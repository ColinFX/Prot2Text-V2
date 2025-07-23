"""
Stage 1 - contrastive learning training script for ESM-C + Qwen modality alignment 
on ESMCQwen model. 

Without LoRA. 

DistributedDataParallel training script implemented from scratch. 

The script currently supports gradient accumulation, AutoMixedPrecision, 
inter-epoch evaluation. 

The script currently does not support save/load pretrained, gradient checkpointing 
or generation under FSDP. 

reference for AMP: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html 

* The script is designed for multi-GPU parallelism on single node.
* On the cluster, print(...) will go to stdout and tqdm(...) will go to stderr.
"""

import argparse
from datetime import datetime
import os
from typing import Any, Dict, Literal, Union

import torch
import torch.utils.data
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel, AutoModelForCausalLM, EsmConfig

from dataset import Prot2TextLightDataset, Prot2TextLightCollater
from models import (
    ModalityAdapter, 
    ModalityAdapterConfig, 
    ESMCConfig,
    ESMCQwen
)
from esm.models.esmc import ESMC
import scripts.utils_argparse as utils_argparse


argParser = argparse.ArgumentParser()

# Required arguments
argParser.add_argument("--root_csv_dir", type=str, required=True, help="Directory containing train/eval CSV files")
argParser.add_argument("--save_checkpoint_dir", type=str, required=True, help="Directory to save checkpoints")

# Optional checkpoint loading
argParser.add_argument("--load_model_checkpoint_path", type=str, default="", help="Path to model checkpoint to load")
argParser.add_argument("--load_optimizer_scheduler_checkpoint_path", type=str, default="", help="Path to optimizer/scheduler checkpoint to load")

# Training hyperparameters with sensible defaults
argParser.add_argument("--torch_dtype", type=utils_argparse.str2dtype, default="float16", help="Training dtype (default: float16)")
argParser.add_argument("--batch_size_per_device", type=int, default=8, help="Batch size per device (default: 8)")
argParser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs (default: 5)")
argParser.add_argument("--save_every_epochs", type=int, default=1, help="Save checkpoint every N epochs (default: 1)")
argParser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps (default: 1)")
argParser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
argParser.add_argument("--gradient_clipping", type=float, default=None, help="Gradient clipping threshold (default: None)")
argParser.add_argument("--scheduler_gamma", type=float, default=0.99, help="Scheduler gamma (default: 0.99)")
argParser.add_argument("--random_seed", type=int, default=42, help="Random seed (default: 42)")
argParser.add_argument("--contrastive_num_segments", type=int, default=2, help="Number of contrastive segments (default: 2)")

# Dataset splits with defaults
argParser.add_argument("--train_split", type=str, default="train", help="Training split name (default: train)")
argParser.add_argument("--eval_split", type=str, default="validation", help="Evaluation split name (default: validation)")
argParser.add_argument("--debug_trim_train_split", type=int, default=None, help="Trim training dataset for debugging")
argParser.add_argument("--debug_trim_eval_split", type=int, default=None, help="Trim evaluation dataset for debugging")


class BatchInfoNCELoss(torch.nn.Module):
    r"""
    Batch version of InfoNCE loss for contrastive learning. Positive and negative 
    pairs are picked within every batch.

    $L_{\text{InfoNCE}} = -\log \frac{\exp(s(x, x^+)/\tau)}{\sum_{x'} \exp(s(x, x')/\tau)}$

    temperature: float, temperature parameter for the softmax function. The smaller
    the temperature, the sharper the softmax distribution. 
    """
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, batch_output1: torch.Tensor, batch_output2: torch.Tensor):
        similarity_matrix = torch.mm(batch_output1, batch_output2.t())
        logits = similarity_matrix / self.temperature  # (bsz, bsz)
        numerator = torch.exp(torch.diag(logits)).unsqueeze(1)  # (bsz, 1)
        denominator = torch.sum(torch.exp(logits), dim=1, keepdim=True)  # (bsz, 1)
        return - torch.log(numerator / denominator).mean()


class SegmentedBatchInfoNCELoss(torch.nn.Module):
    """Segmented version of BatchInfoNCELoss for contrastive learning."""
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
            self, 
            segment_output1: torch.Tensor,  # (segment_size, hidden_dim)
            batch_output2: torch.Tensor,  # (bsz, hidden_dim)
            labels: torch.Tensor  # (segment_size,)
        ):
        segment_size = segment_output1.size(0)

        similarity_matrix = torch.mm(segment_output1, batch_output2.t())
        logits = similarity_matrix / self.temperature  # (segment_size, bsz)
        
        numerator = torch.exp(logits[torch.arange(segment_size), labels]).unsqueeze(1)  # (segment_size, 1)
        denominator = torch.sum(torch.exp(logits), dim=1, keepdim=True)  # (segment_size, 1)
        
        return - torch.log(numerator / denominator).mean()


def load_model(args: Dict[str, Any]) -> PreTrainedModel:
    """
    Standard API for different models. Used in both `train` and `generate`.
    Load base model of the given name, and load weights from the checkpoint path 
    if provided.

    Returned model should be on CPU and under default data type.
    A general checkpoint shall contain the model state dict, optimizer state dict,
    and scheduler state dict.
    """
    esm_encoder = ESMC.from_pretrained(ESMCConfig.esm_model_name)

    llm_decoder = AutoModelForCausalLM.from_pretrained(
        ESMCConfig.llm_model_name,
        trust_remote_code=True,
        torch_dtype=args["torch_dtype"],
        device_map="auto",
    )

    adapter_config = ModalityAdapterConfig(
        input_dim=esm_encoder.embed.embedding_dim,
        intermediate_dim=2048,
        output_dim=llm_decoder.config.hidden_size,
    )
    adapter = ModalityAdapter(adapter_config)
    adapter.to(args["torch_dtype"])
    model_cfg = ESMCConfig(
        adapter_config=adapter_config,
        llm_config=llm_decoder.config,
    )
    
    model = ESMCQwen(
        config=model_cfg,
        esm_encoder=esm_encoder,
        adapter=adapter,
        llm_decoder=llm_decoder,
    )

    if args["load_model_checkpoint_path"]:
        print(f"Loading {args['load_model_checkpoint_path']}")
        adapter_state_dict = torch.load(
            args["load_model_checkpoint_path"], 
            weights_only=True, 
            map_location="auto",
            low_cpu_mem_usage=True
        )
        model.adapter.load_state_dict(adapter_state_dict)

    # WARNING: esm and llm weights are fixed
    model.esm_encoder.requires_grad_(False)
    model.llm_decoder.requires_grad_(False)

    return model


def readout_embeddings(
        embeddings: torch.Tensor,  # (bsz, seq_len, hidden_dim)
        attention_mask: torch.Tensor,  # (bsz, text_len)
        readout_fn: Literal["last", "mean", "std", "mix"], 
) -> torch.Tensor:
    """
    Perform a readout operation on the output sequence embeddings of the forward 
    pass, given the attention mask. 
    """
    # Debug: Check input tensors
    print(f"DEBUG READOUT: embeddings.shape={embeddings.shape}, has_nan={torch.isnan(embeddings).any()}")
    print(f"DEBUG READOUT: attention_mask.shape={attention_mask.shape}, has_nan={torch.isnan(attention_mask.float()).any()}")
    
    if readout_fn == "last":
        # inputs must be right padded
        # for left padding simply take the last token and do not use this function
        last_token_indices = attention_mask.sum(dim=1) - 1  # (bsz,)
        batch_indices = torch.arange(
            attention_mask.size(0), 
            device=attention_mask.device
        )  # (bsz,)
        result = embeddings[batch_indices, last_token_indices, :]  # (bsz, hidden_dim)
        print(f"DEBUG READOUT: last result has_nan={torch.isnan(result).any()}")
        return result

    elif readout_fn == "mean":
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        print(f"DEBUG READOUT: masked_embeddings has_nan={torch.isnan(masked_embeddings).any()}")
        
        sum_embeddings = masked_embeddings.sum(dim=1)  # (bsz, hidden_dim)
        print(f"DEBUG READOUT: sum_embeddings has_nan={torch.isnan(sum_embeddings).any()}")
        
        count_attn_mask = attention_mask.sum(dim=1, keepdim=True)  # (bsz, 1)
        print(f"DEBUG READOUT: count_attn_mask={count_attn_mask.squeeze()}")
        
        # Prevent division by zero
        result = sum_embeddings / count_attn_mask.clamp(min=1)  # (bsz, hidden_dim)
        print(f"DEBUG READOUT: mean result has_nan={torch.isnan(result).any()}")
        return result
    
    elif readout_fn == "std":
        mean_embeddings = readout_embeddings(
            embeddings=embeddings, 
            attention_mask=attention_mask, 
            readout_fn="mean"
        )  # (bsz, hidden_dim)
        print(f"DEBUG READOUT: mean_embeddings for std has_nan={torch.isnan(mean_embeddings).any()}")
        
        diff_embeddings = embeddings - mean_embeddings.unsqueeze(1)
            # (bsz, text_len, hidden_dim)
        print(f"DEBUG READOUT: diff_embeddings has_nan={torch.isnan(diff_embeddings).any()}")
        
        diff_embeddings_2 = diff_embeddings.pow(2) 
        print(f"DEBUG READOUT: diff_embeddings_2 has_nan={torch.isnan(diff_embeddings_2).any()}")
        
        masked_diff_embeddings_2 = diff_embeddings_2 * attention_mask.unsqueeze(-1)
        print(f"DEBUG READOUT: masked_diff_embeddings_2 has_nan={torch.isnan(masked_diff_embeddings_2).any()}")
        
        sum_diff_embeddings_2 = masked_diff_embeddings_2.sum(dim=1)  # (bsz, hidden_dim)
        print(f"DEBUG READOUT: sum_diff_embeddings_2 has_nan={torch.isnan(sum_diff_embeddings_2).any()}")
        
        count_attn_mask = attention_mask.sum(dim=1, keepdim=True)  # (bsz, 1)
        print(f"DEBUG READOUT: count_attn_mask for std={count_attn_mask.squeeze()}")
        
        # Add small epsilon to prevent sqrt(0) and division by 0
        variance = sum_diff_embeddings_2 / count_attn_mask.clamp(min=1)
        print(f"DEBUG READOUT: variance has_nan={torch.isnan(variance).any()}")
        
        result = (variance + 1e-8).sqrt()  # (bsz, hidden_dim)
        print(f"DEBUG READOUT: std result has_nan={torch.isnan(result).any()}")
        return result

    elif readout_fn == "mix": 
        print("DEBUG READOUT: Computing mix (mean + std)")
        mean_embeddings = readout_embeddings(
            embeddings=embeddings, 
            attention_mask=attention_mask, 
            readout_fn="mean"
        )
        print(f"DEBUG READOUT: mean_embeddings for mix has_nan={torch.isnan(mean_embeddings).any()}")
        
        std_embeddings = readout_embeddings(
            embeddings=embeddings, 
            attention_mask=attention_mask, 
            readout_fn="std"
        )
        print(f"DEBUG READOUT: std_embeddings for mix has_nan={torch.isnan(std_embeddings).any()}")
        
        result = torch.cat([mean_embeddings, std_embeddings], dim=1)  # (bsz, 2 * hidden_dim)
        print(f"DEBUG READOUT: mix result has_nan={torch.isnan(result).any()}")
        return result


def get_sequence_embeddings(
        model: ESMCQwen, 
        protein_sequences: list,
) -> torch.Tensor:
    """
    Take mean pooling and the std pooling of the adapter outputs for each valid 
    token in the sequence as the sequence embedding for contrastive learning. 

    protein_sequences: List of raw protein sequences
    return: (bsz, decoder_hidden_size)
    """
    encoder_output = model.forward(
        protein_sequences=protein_sequences,
        return_encoder_outputs=True,
    )

    adapter_output = encoder_output[0]  # (bsz, max_seq_len, decoder_hidden_size)
    print(f"DEBUG SEQUENCE: adapter_output.shape={adapter_output.shape}, has_nan={torch.isnan(adapter_output).any()}")
    
    # Create attention mask based on sequence lengths (all positions are valid for raw sequences)
    batch_size, seq_len = adapter_output.shape[:2]
    attention_mask = torch.ones(
        (batch_size, seq_len), 
        dtype=torch.long, 
        device=adapter_output.device
    )
    print(f"DEBUG SEQUENCE: attention_mask.shape={attention_mask.shape}")

    result = readout_embeddings(
        embeddings=adapter_output, 
        attention_mask=attention_mask, 
        readout_fn="mix"
    )  # (bsz, decoder_hidden_size)
    print(f"DEBUG SEQUENCE: final result has_nan={torch.isnan(result).any()}")
    
    return result


def get_description_embeddings(
        model: ESMCQwen,
        description_input_ids: torch.Tensor,
        description_attention_mask: torch.Tensor,
        output_llm_layer: int = 16,
) -> torch.Tensor:
    """Extract embeddings from the Qwen 14B decoder for description text."""
    print(f"DEBUG DESCRIPTION: Starting get_description_embeddings")
    
    with torch.no_grad():  # WARNING: llm decoder fixed during contrastive training
        qwen_model = model.llm_decoder.model
        
        outputs = qwen_model(
            input_ids=description_input_ids,
            attention_mask=description_attention_mask,
            use_cache=False, 
            output_attentions=False, 
            output_hidden_states=True, 
            return_dict=True,
        )
        
        # Extract hidden states from the specified layer
        hidden_states = outputs.hidden_states[output_llm_layer]
        print(f"DEBUG DESCRIPTION: hidden_states.shape={hidden_states.shape}, has_nan={torch.isnan(hidden_states).any()}")

    result = readout_embeddings(
        embeddings=hidden_states,
        attention_mask=description_attention_mask,
        readout_fn="mix"
    )  # (bsz, decoder_hidden_size)
    
    print(f"DEBUG DESCRIPTION: final result has_nan={torch.isnan(result).any()}")
    return result


def teacher_forcing_forward_pass(
        rank: int,
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        data_batch: Dict[str, Any],
        contrastive_num_segments: int, 
) -> torch.Tensor:  # loss
    """
    Standard API for different models. Used in both `train_epoch` and `eval_epoch`.
    Prepare inputs from dataloader, migrate variable to the same device as the model, 
    and execute the forward pass with teacher forcing.

    Returned loss is not scaled with gradient accumulation steps.

    Due to the memory limit on GPUs, the similarity matrix will be computed in 
    segments, and the loss will be averaged over the segments.
    """
    protein_sequences = data_batch["protein_sequences"]  # List of raw sequences
    description_input_ids = data_batch["description_input_ids"].to(rank)
    description_attention_mask = data_batch["description_attention_mask"].to(rank)

    base_model = model
    if isinstance(model, DistributedDataParallel):
        base_model = model.module

    batch_size = len(protein_sequences)
    segment_size = batch_size // contrastive_num_segments
    if segment_size * contrastive_num_segments != batch_size:
        print(
            "WARNING: Given batch size is not divisible by the number of segments "
            "for contrastive learning."
        )
    
    acc_loss = torch.zeros([]).to(rank)
    loss_fn = SegmentedBatchInfoNCELoss()

    # Debug: Print batch info
    print(f"DEBUG: batch_size={batch_size}, segment_size={segment_size}, contrastive_num_segments={contrastive_num_segments}")

    with torch.no_grad():  # WARNING: llama decoder fixed
        description_output = get_description_embeddings(
            base_model,  
            description_input_ids, 
            description_attention_mask
        )
        description_output = torch.nn.functional.normalize(description_output, p=2, dim=-1)

    for segment_id in range(contrastive_num_segments):
        segment_protein_sequences = protein_sequences[
            segment_id * segment_size:(segment_id + 1) * segment_size
        ]

        segment_protein_output = get_sequence_embeddings(
            base_model, 
            segment_protein_sequences
        )
        segment_protein_output = torch.nn.functional.normalize(segment_protein_output, p=2, dim=-1)
        
        # Debug: Check tensor shapes and values
        print(f"DEBUG: segment_id={segment_id}, segment_protein_output.shape={segment_protein_output.shape}")
        print(f"DEBUG: description_output.shape={description_output.shape}")
        print(f"DEBUG: segment_protein_output has NaN: {torch.isnan(segment_protein_output).any()}")
        print(f"DEBUG: description_output has NaN: {torch.isnan(description_output).any()}")
        
        labels = torch.arange(
            segment_id * segment_size, 
            (segment_id + 1) * segment_size, 
            device=rank
        )
        
        print(f"DEBUG: labels={labels}")

        segment_loss = loss_fn(
            segment_output1=segment_protein_output, 
            batch_output2=description_output, 
            labels=labels
        )
        
        print(f"DEBUG: segment_loss={segment_loss.item()}")
        acc_loss += segment_loss

    return acc_loss / contrastive_num_segments


def setup(rank: int, world_size: int):
    """
    Initialize processes for distributed training before first epoch. 
    Fetch from job script or launcher to set the IP address and the port of the 
    master node. 
    """
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '9901')
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """End processes for distributed training after last epoch"""
    dist.destroy_process_group()


def train_epoch(
        rank: int,
        current_epoch: int,
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        dataloader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        args: Dict[str, Any]
):
    """Iterate over all batches for one epoch in training with teacher forcing"""
    model.train()
    ddp_loss = torch.zeros(2).to(rank)  
        # [0] for acc. loss and [1] for num. of seen batches
    ddp_gradnorm = torch.zeros(2).to(rank)  
        # [0] for acc. gradnorm and [1] for num. of passed steps
    optimizer.zero_grad()  # erase accumulated gradients from last epoch

    t = tqdm(iter(dataloader))
    for batch_idx, data_batch in enumerate(t):
        # with autocast, logits will be in AUTOCAST_DTYPE 
        # but loss will be re-casted to torch.float32
        # and model weights will stay in torch.float32
        loss = teacher_forcing_forward_pass(
            rank=rank, 
            model=model, 
            data_batch=data_batch, 
            contrastive_num_segments=args["contrastive_num_segments"]
        )

        # rescale loss for consistency with different gradient accumulation steps
        loss = loss / args["gradient_accumulation_steps"]

        # summary current batch
        t.set_postfix({
            "mode": "train",
            "epoch": f"{current_epoch}/{args['num_epochs']}",
            "batch_loss": loss.item() * args["gradient_accumulation_steps"],
            "device": f"rank:{rank}"
        })
        ddp_loss[0] += loss.item() * args["gradient_accumulation_steps"]
        ddp_loss[1] += 1  # the loss is the weighted mean of the output of every batch

        # scale the loss up by a large factor to prevent them from becoming too small
        # then accumulate the scaled grads
        loss.backward()  
            # backward out of autocast, but still uses same dtype as for forward

        # update weights by loss if accumulation step is reached
        if (batch_idx + 1) % args["gradient_accumulation_steps"] == 0: 
            gradnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(
                    float("inf") 
                    if args["gradient_clipping"] is None 
                    else args["gradient_clipping"]
                )
            )
            ddp_gradnorm[0] += gradnorm
            ddp_gradnorm[1] += 1

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # summary current epoch
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            f"[epoch={current_epoch}/{args['num_epochs']}, "
            f"train_loss={ddp_loss[0] / ddp_loss[1]}, "
            f"epoch_lr={optimizer.param_groups[0]['lr']}, "
            f"epoch_gradnorm={ddp_gradnorm[0] / ddp_gradnorm[1]}]"
        )
        # NaN detection
        if ddp_loss[0] != ddp_loss[0]:
            raise ValueError(
                "NaN detected in the training loss of the epoch, training interrupted."
            )


def eval_epoch(
        rank: int,
        current_epoch: int, 
        model: Union[DistributedDataParallel, FullyShardedDataParallel],
        dataloader: torch.utils.data.DataLoader,
        args: Dict[str, Any]
):
    """Iterate over all batches in evaluation with teacher forcing"""
    model.eval()
    ddp_loss = torch.zeros(2).to(rank)  
        # [0] for acc. loss and [1] for num. of seen batches

    t = tqdm(iter(dataloader))
    for data_batch in t:
        with torch.no_grad():
            loss = teacher_forcing_forward_pass(
                rank=rank,
                model=model,
                data_batch=data_batch,
                contrastive_num_segments=args["contrastive_num_segments"]
            )

            t.set_postfix({
                "mode": "eval",
                "epoch": f"{current_epoch}/{args['num_epochs']}",
                "batch_loss": loss.item(),
                "device": f"rank:{rank}"
            })
            ddp_loss[0] += loss.item()
            ddp_loss[1] += 1

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            f"[epoch={current_epoch}/{args['num_epochs']}, "
            f"eval_loss={ddp_loss[0] / ddp_loss[1]}]"
        )


def train_on_device(
        rank: int,
        world_size: int,
        args: Dict[str, Any]
):
    """
    Training and evaluation process for each device, including epochs of training 
    with teacher forcing. 
    """
    setup(rank, world_size)

    # prepare datasets and dataloaders
    # For ESMCQwen: 
    # - ESMC handles raw protein sequences (no tokenizer needed)
    # - Qwen 14B handles text (use AutoTokenizer)
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        ESMCConfig.llm_model_name,  # "Qwen/Qwen3-14B"
        pad_token='<|reserved_special_token_0|>'
    )

    train_dataset = Prot2TextLightDataset(
        csv_path=os.path.join(args["root_csv_dir"], f"{args['train_split']}.csv")
    )
    if args["debug_trim_train_split"]:
        # Trim the dataset for debugging
        original_data = train_dataset.data
        train_dataset.data = original_data[:args["debug_trim_train_split"]]
    
    train_sampler = DistributedSampler(
        train_dataset, 
        rank=rank, 
        num_replicas=world_size, 
        shuffle=True
    )
    
    train_collater = Prot2TextLightCollater(
        sequence_tokenizer=None,  # Not needed for raw sequences
        description_tokenizer=qwen_tokenizer,  # Qwen tokenizer for text
        mode="train",
        include_text_fields=True,
        use_raw_sequences=True,  # Use raw sequences for ESMCQwen
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["batch_size_per_device"],
        sampler=train_sampler,
        num_workers=4,  # parallel CPU cores used for data loading
        pin_memory=True,  # enable page-locked memory allocation for faster data transfer to GPUs
        shuffle=False,  # avoid shuffling twice with DistributedSampler
        drop_last=True,  # avoid incomplete batch at the end
        collate_fn=train_collater,
    )
    print(f"Train dataset loaded on rank:{rank}")

    eval_dataset = Prot2TextLightDataset(
        csv_path=os.path.join(args["root_csv_dir"], f"{args['eval_split']}.csv")
    )
    if args["debug_trim_eval_split"]:
        # Trim the dataset for debugging
        original_data = eval_dataset.data
        eval_dataset.data = original_data[:args["debug_trim_eval_split"]]
    
    eval_sampler = DistributedSampler(
        eval_dataset, 
        rank=rank, 
        num_replicas=world_size, 
        shuffle=False
    )
    
    eval_collater = Prot2TextLightCollater(
        sequence_tokenizer=None,  # Not needed for raw sequences
        description_tokenizer=qwen_tokenizer,  # Qwen tokenizer for text
        mode="train",  # Use train mode for contrastive learning (need both sequences and descriptions)
        include_text_fields=True,
        use_raw_sequences=True,  # Use raw sequences for ESMCQwen
    )
    
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args["batch_size_per_device"],
        sampler=eval_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        collate_fn=eval_collater,
    )
    print(f"Eval dataset loaded on rank:{rank}")

    torch.cuda.set_device(rank)

    model = load_model(args=args)
    model = model.to(rank)

    model = DistributedDataParallel(
        model, 
        find_unused_parameters=True  # suppress error for unused parameters in wrapped model
    )
    print(f"DDP model loaded on rank:{rank}")

    # initialization of the optimizer after wrapping the model
    optimizer = Adam(model.parameters(), lr=args["learning_rate"])
    scheduler = StepLR(optimizer, step_size=1, gamma=args["scheduler_gamma"])
    if args["load_optimizer_scheduler_checkpoint_path"]:
        print(f"Loading {args['load_optimizer_scheduler_checkpoint_path']}")
        checkpoint_state_dicts = torch.load(
            args["load_optimizer_scheduler_checkpoint_path"], 
            weights_only=True
        )
        optimizer_state_dict = checkpoint_state_dicts["optimizer_state_dict"]
        scheduler_state_dict = checkpoint_state_dicts["scheduler_state_dict"]
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)

    # core loop of epochs
    for epoch_idx in range(1, args["num_epochs"] + 1):
        # shuffle data differently at each epoch across all processes
        train_sampler.set_epoch(epoch=epoch_idx)

        train_epoch(
            rank=rank,
            current_epoch=epoch_idx,
            model=model,    
            dataloader=train_loader,
            optimizer=optimizer,
            args=args
        )
        scheduler.step()
        dist.barrier()  # use a barrier to make sure training is done on all ranks
        
        eval_epoch(
            rank=rank,
            model=model,
            current_epoch=epoch_idx,
            dataloader=eval_loader,
            args=args
        )
        dist.barrier()

        if (
            epoch_idx == 1 
            or epoch_idx == args["num_epochs"] 
            or epoch_idx % args["save_every_epochs"] == 0
        ):
            adapter_state_dict = model.module.adapter.state_dict()
            if rank == 0:
                model_checkpoint_path = os.path.join(
                    args["save_checkpoint_dir"], 
                    f"model_checkpoint_{epoch_idx}.pt"
                )
                torch.save(adapter_state_dict, model_checkpoint_path)
                print(f"Saving {model_checkpoint_path}")

                optimizer_scheduler_checkpoint_path = os.path.join(
                    args["save_checkpoint_dir"], 
                    f"optimizer_scheduler_checkpoint_{epoch_idx}.pt"
                )
                torch.save(
                    {
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    }, 
                    optimizer_scheduler_checkpoint_path
                )
                print(f"Saving {optimizer_scheduler_checkpoint_path}")

            dist.barrier()

    cleanup()


def train_distributed(
        args: Dict[str, Any]  # replace **kwargs for compatibility with spawn
):
    """
    Core training process across multiple devices with epochs of training and 
    inter-epoch evaluation.
    """
    torch.multiprocessing.spawn(
        train_on_device, 
        args=(args["world_size"], args),
        nprocs=args["world_size"],
        join=True
    )


if __name__ == '__main__':
    # suppress messages from AutoTokenizer parallelism and Graphein respectively
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["LOGURU_LEVEL"] = "INFO"

    parsed_args = argParser.parse_args()
    
    # Print help information about model paths
    print("####################")
    print("ESMCQwen Contrastive Training")
    print(f"Using ESM model: {ESMCConfig.esm_model_name}")
    print(f"Using LLM model: {ESMCConfig.llm_model_name}")
    print("####################")
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # restrict GPU visibility
    parsed_args.world_size = torch.cuda.device_count()  # use up all visible GPUs

    torch.manual_seed(parsed_args.random_seed)
    torch.cuda.manual_seed(parsed_args.random_seed)
    
    # initialize checkpoint directory
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    parsed_args.save_checkpoint_dir = os.path.join(
        parsed_args.save_checkpoint_dir, 
        f"checkpoints_{timestamp}"
    )
    if not os.path.exists(parsed_args.save_checkpoint_dir):
        os.mkdir(parsed_args.save_checkpoint_dir)
    
    print("####################")
    print("Training Configuration:")
    for key, value in parsed_args.__dict__.items(): 
        print(f"{key}: {value}")
    print("####################")

    train_distributed(parsed_args.__dict__)
