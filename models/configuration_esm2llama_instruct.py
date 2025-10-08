"""
Configuration class for the assembled Esm2LlamaInstructForCausalLM model. 

Esm2LlamaInstructConfig = EsmConfig + ModalityAdapterConfig + LlamaConfig
"""

from typing import Dict, Optional, Union
from transformers import EsmConfig, LlamaConfig, PretrainedConfig


class ModalityAdapterConfig(PretrainedConfig):
    """Configuration class of the 2-layer non-linear adapter."""
    model_type = "modality_adapter"  # unique identifier of the model

    def __init__(
            self, 
            input_dim: int, 
            intermediate_dim: int,
            output_dim: int, 
            dropout_rate: float = 0.3,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate


class Esm2LlamaInstructConfig(PretrainedConfig):
    """
    Configuration class of Esm2LlamaInstructForCausalLM model.
    placeholder_id: Token id in chat template to be replaced by ESM embeddings.
    """
    model_type = "esm2llama_instruct"  # unique identifier of the model

    def __init__(
            self, 
            # model components
            esm_config: Optional[Union[EsmConfig, Dict]] = None, 
            adapter_config: Optional[Union[ModalityAdapterConfig, Dict]] = None,
            llama_config: Optional[Union[LlamaConfig, Dict]] = None, 
            # standalone attributes
            placeholder_id: int = 128003, 
            **kwargs
    ):
        super().__init__(**kwargs)
        
        if isinstance(esm_config, dict):
            self.esm_config = EsmConfig(**esm_config)
        else:
            self.esm_config = esm_config
            
        if isinstance(llama_config, dict):
            self.llama_config = LlamaConfig(**llama_config)
        else:
            self.llama_config = llama_config
            
        if isinstance(adapter_config, dict):
            self.adapter_config = ModalityAdapterConfig(**adapter_config)
        else:
            self.adapter_config = adapter_config
            
        self.placeholder_id = placeholder_id
