# Prot2Text-V2: Protein Function Prediction with Multimodal Contrastive Alignment

<div align="center">
<img src="https://img.shields.io/badge/--Transformers-ff9d0b.svg" alt="transformers">
<img src="https://img.shields.io/badge/Base-ESM--2-0082fb.svg" alt="based-on-esm">
<img src="https://img.shields.io/badge/Base-LLAMA--3.1-0082fb.svg" alt="based-on-llama">
<img src="https://img.shields.io/badge/--AI4Biology-green.svg" alt="ai4biology">
<img src="https://img.shields.io/badge/License-MIT-darkgrey.svg" alt="license-mit">
</div>

<div align="center">
| <a href="https://arxiv.org/abs/2505.11194">📃 Preprint</a> |
<a href="https://neurips.cc/virtual/2025/poster/115368">📜 NeurIPS</a> |
<a href="https://huggingface.co/xiao-fei/Prot2Text-V2-11B-Instruct-hf">🤗 Model</a> |
<a href="https://huggingface.co/datasets/habdine/Prot2Text-Data">🤗 Dataset</a> |
</div>

<br>

This is the official repository for the paper "**Prot2Text-V2: Protein Function Prediction with Multimodal Contrastive Alignment**" by Xiao Fei, Michail Chatzianastasis, Sarah Almeida Carneiro, Hadi Abdine, Lawrence P. Petalidis, and Michalis Vazirgiannis.

We're excited to share that our paper has been accepted to **NeurIPS 2025**! The pretrained model weights and the dataset are now publicly available on Hugging Face.

## About the Project

Proteins are written in a code made of amino acids, but what if we could actually read that code like a language? 

**Prot2Text-V2** treats a protein sequence as if it were another language, and then translate it into English. The model takes the raw amino acid sequence as inpout and generates a clear, human-readable paragraph describing what the protein does. 

<div align="center">
<img src="./figures/model.png" alt="Model Architecture" width="95%"/>
</div>

The instruction-based Prot2Text-V2 model is an innovative fusion of three key components: 

* Protein language model as sequence encoder: `facebook/esm2_t36_3B_UR50D`
* Modality adapter as a unique and lightweight component that bridges the gap between protein embeddings and the language model. 
* Natural language decoder for generating articulate textual descriptions utilizing the sequence embeddings: `meta-llama/Llama-3.1-8B-Instruct`

<div align="center">
<img src="./figures/training.png" alt="Training Stages" width="85%"/>
</div>

A clever alignment step first captures the semantic meaning of the sequence, after which  supervised fine-tuning trains the decoder to generate articulate descriptions. 

For backward compatibility, the repository also includes our legacy base model, `Esm2LlamaForCausalLM`, along with its specialized dataloader.

## Getting Started

✅ Verified on Ubuntu-22.04-LTS with 2 x NVIDIA RTX A6000

✅ Verified on RHEL-9.4 with 8 x NVIDIA A100 

* Install NVIDIA `cuda-toolkit=12.1.1`, see official website for detailed information. 

* Install `dssp=4.0.4` for protein dataset preprocessing: 

    ```shell
    sudo apt-get install dssp=4.0.4
    ```

* Create environment with `conda` then install packages with `pip`: 

    ```shell
    conda create -n prot2text-pip python=3.8
    
    pip3 install torch torchvision torchaudio  # torch==2.3.0
    pip3 install torch_geometric
    pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
    
    pip3 install graphein==1.7.7

    pip3 install transformers==4.40.2 tokenizers==0.19.1 accelerate==0.29.3 sentencepiece==0.2.0
    pip3 install peft==0.10.0
    pip3 install biopython==1.81
    pip3 install networkx==2.5
    pip3 install chardet==5.2.0 charset-normalizer==2.0.4
    pip3 install multiprocess==0.70.16
    pip3 install tensorboard==2.14.0
    pip3 install evaluate==0.4.2
    pip3 install mpi4py==3.1.6
    
    sudo apt install libaio-dev
    DS_BUILD_FUSED_ADAM pip3 install deepspeed==0.14.2
    
    pip3 install nltk==3.8.1 rouge_score==0.1.2 jiwer==3.0.4
    ```

## Repository Outline

### `/dataset` (Dataset Handling)

* `./dataset.py`: [`Prot2TextInstructDataset`]

    * Prepares the SwissProt dataset (a curated protein database) for training using an instruction-based model.
    * Loads the preprocessed dataset for use with either the instruction-based model or the standard (base) model.

* `./dataloader.py`: [`Prot2TextInstructDataLoader`]
    
    * Works alongside `Prot2TextInstructDataset` to handle data loading.
    * Formats data into batches using a chat-style template, making it suitable for instruction-based models.

### `/models` (Model Configuration and Architecture)

* `./configuration_esm2llama_instruct.py`: [`Esm2LlamaInstructConfig`]

    * Defines the standard configuration settings for the instruction-based model.

* `./modeling_esm2llama_instruct.py`: [`Esm2LlamaInstructForCausalLM`]

    * Implements the actual instruction-based model, using the configuration defined in `Esm2LlamaInstructConfig`.

* `./configuration_esm2llama_legacy.py`: [`Esm2LlamaConfig`]

    * Defines the standard configuration settings for the legacy base model (non-instruction-based).

* `./modeling_esm2llama_legacy.py`: [`Esm2LlamaForCausalLM`]

    * Implements the legacy base model, using the configuration defined in `Esm2LlamaConfig`.

### `/scripts` (Training and Evaluation Scripts)

* `./train_contrast.py`: 

    * Contrastive Learning - Stage 1 of instruction model training. 
    * Trains the instruction-based model by learning to differentiate between good and bad outputs.

* `./train_instruct.py`: 

    * Instruction Tuning - Stage 2 of instruction model training. 
    * Further trains the instruction-based model by fine-tuning it on instruction-response data.

* `./train_legacy.py`: 

    * Supervised Fine-Tuning for the legacy Model. 
    * Trains the legacy base model using standard supervised learning techniques.


## Dataset Preparation

* Download CSV files from [HuggingFace](https://huggingface.co/datasets/habdine/Prot2Text-Data) and place under `./data`.  

* Download PDB files from AlphaFoldDB (for RGCN only) then preprocess graph and text features: 

```python
from transformers import AutoTokenizer
from dataset import Prot2TextInstructDataset

SPLIT = "train"  # run script for "eval" and "test" as well
CSV_DIR = "./data"
DATA_ROOT_DIR = "/data/Prot2Text-Llama3-Data"
LLAMA_DIR = "/data/Meta-Llama-3.1-8B-Instruct-hf"
ESM_DIR = "/datadisk/esm2_t36_3B_UR50D"

split_dataset = Prot2TextInstructDataset(
    root_dir=os.path.join(DATA_ROOT_DIR, SPLIT),
    csv_path=os.path.join(CSV_DIR, f"{SPLIT}.csv"),
    sequence_tokenizer=AutoTokenizer.from_pretrained(ESM_DIR),
    description_tokenizer=AutoTokenizer.from_pretrained(LLAMA_DIR, pad_token='<|reserved_special_token_0|>'),
    skip_download=False,
    skip_reload=False, 
)
```

* [Optional] In case of applying new language tokenizer to a preprocessed dataset, run the following to avoid processing graphs again: 

```python
NEW_LLAMA_DIR = "/data/Llama-3.2-1B"

split_dataset = Prot2TextInstructDataset(
    root_dir=os.path.join(DATA_ROOT_DIR, SPLIT),
    csv_path=os.path.join(CSV_DIR, f"{SPLIT}.csv"),
    sequence_tokenizer=AutoTokenizer.from_pretrained(ESM_DIR),
    description_tokenizer=AutoTokenizer.from_pretrained(NEW_LLAMA_DIR, pad_token='<|reserved_special_token_0|>'),
    skip_download=True,
    skip_reload=True, 
)
split_dataset.process_text()
```

## Model Training Pipeline

### 1. Contrastive Learning Stage
`./scripts/train_contrast.py` performs contrastive learning to align protein representations with textual descriptions. This stage helps the model learn meaningful cross-modal embeddings.

**Arguments:**
- **Model Paths:**
  - `--esm_path`: Path to pretrained ESM protein language model
  - `--llama_path`: Path to pretrained LLaMA language model
- **Data Directories:**
  - `--root_dataset_dir`: Root directory containing protein datasets
  - `--root_csv_dir`: Directory containing CSV metadata files
- **Checkpoint Handling:**
  - `--save_checkpoint_dir`: Directory to save model checkpoints
  - `--load_model_checkpoint_path`: Path to load full model checkpoint (optional)
  - `--load_optimizer_scheduler_checkpoint_path`: Path to load optimizer/scheduler state (optional)
- **Training Parameters:**
  - `--torch_dtype`: PyTorch data type for training (e.g., float16, float32)
  - `--batch_size_per_device`: Batch size per GPU/device
  - `--num_epochs`: Total number of training epochs
  - `--save_every_epochs`: Frequency of checkpoint saving (in epochs)
  - `--gradient_accumulation_steps`: Number of steps for gradient accumulation
  - `--learning_rate`: Initial learning rate
  - `--gradient_clipping`: Gradient clipping value (optional)
  - `--scheduler_gamma`: Learning rate scheduler gamma value
  - `--random_seed`: Random seed for reproducibility
  - `--contrastive_num_segments`: Number of segments for contrastive learning
- **Data Splits:**
  - `--train_split`: Name of training split
  - `--eval_split`: Name of evaluation split
  - `--debug_trim_train_split`: Trim training set for sanity check (optional)
  - `--debug_trim_eval_split`: Trim evaluation set for sanity check (optional)

### 2. Supervised Fine-Tuning Stage
After contrastive learning, run `./scripts/train_instruct.py` for instruction fine-tuning on the training set.

**Additional/Modified Arguments:**
- **Adapter Configuration:**
  - `--load_adapter_checkpoint_dir`: Directory to load adapter checkpoints
  - `--fix_modality_adapter`: Whether to freeze modality adapter weights
  - `--lora_rank`: Rank for LoRA adapter layers
- **Text Field Handling:**
  - `--include_text_fields`: Whether to include text fields in input
  - `--name_dropout`: Dropout rate for protein names
  - `--taxonomy_dropout`: Dropout rate for taxonomy information

## Performance Evaluation

### 1. Generation (`generate_instruct.py`)
Generates answers for proteins in the test set using a trained model.

**Key Arguments:**
- **Generation Parameters:**
  - `--max_generation_length`: Maximum length of generated text
  - `--num_beams`: Number of beams for beam search
  - `--temperature`: Sampling temperature
  - `--do_sample`: Whether to use sampling
  - `--top_p`: Nucleus sampling probability
  - `--top_k`: Top-k sampling value
- **Output Control:**
  - `--save_generation_postfix_identifier`: Identifier for output files
  - `--max_sequence_length`: Maximum input sequence length

### 2. Benchmarking (`benchmark.py`)
Evaluates generated outputs using various metrics.

**Evaluation Options:**
- `--evaluate_exact_match`: Compute exact match accuracy
- `--evaluate_bleu`: Compute BLEU scores
- `--evaluate_rouge`: Compute ROUGE scores
- `--evaluate_bert_score`: Compute BERTScore
- `--read_file_identifier`: Filter generated files by this identifier
- `--verbose`: Print detailed evaluation results

## Usage Notes:
1. For the full training pipeline, first run `train_contrast.py`, then `train_instruct.py`
2. Generation should use the same data splits used during evaluation
3. Benchmarking can be customized to compute only relevant metrics
4. Debug arguments allow for faster iteration during development

The pipeline supports both full fine-tuning and parameter-efficient approaches (LoRA, adapter layers) through the various adapter-related arguments.
