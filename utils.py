"""Utilities functions such as command line parsers."""
from dataclasses import dataclass, field
from typing import Optional, List
import math

from transformers import TrainingArguments as HfTrainingArguments


UNSET_WARMUP = 2**30
UNSET_INIT = UNSET_WARMUP

@dataclass
class TrainingArguments(HfTrainingArguments):
    max_steps: int = field(default=20000)
    warmup_ratio: float = field(default=0.04)
    warmup_steps: int = field(default=UNSET_WARMUP)
    
    min_lr_ratio: float = field(
        default=0.0
    )
    weight_decay: float = field(
        default=0.01
    )
    shaped_attention: str = field(
        default="mixing",
        metadata={
            "help": (
                "Can be shaped, mixing, or vanilla."
            )
        },
    )
    weight_frozen: bool = field(
        default=False,
        metadata={
            "help": (
                "ONLY for the algorithmic tasks."
            )
        },
    )
    ordered: bool = field(
        default=False
    )
    do_rope: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to do rope position embedding."
            )
        },
    )
    per_device_train_batch_size: int = field(
        default=32
    )
    per_device_eval_batch_size: int = field(
        default=32
    )
    num_train_epochs: int = field(
        # E.g. wikitext-103 has 103 million tokens, so training
        # on 10B tokens means 100 epochs.
        default=100
    )
    logging_steps: int = field(
        default=1000
    )
    eval_steps: int = field(
        default=1000
    )
    learning_rate: float = field(
        default=3e-3
    )
    
    freeze_attention: bool = field(
         default=False
    )
    
    freeze_mlp: bool = field(
         default=False
    )
    
    plot_attention_maps: bool = field(
         default=False
    )
    
    scale_lr: bool = field(
        default=False
    )
    
    def __post_init__(self):
        super().__post_init__()
        if self.warmup_steps == UNSET_WARMUP:
            steps = int(self.warmup_ratio * self.max_steps)
            min_steps = 500
            self.warmup_steps = max(steps, min_steps)


@dataclass
class ScriptArguments:
    
    initializer_range: float = field(default=0.02) # TODO make sure it's 1/sqrt(width)
    
    # if self.initializer_range == UNSET_INIT:
    #     self.initializer_range = 1/math.sqrt(self.n_embd)

    model_name_or_path: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": (
                "Model name or path, can be e.g. gpt2, llama."
            )
        },
    )
    
    base_hidden_size: int = field(default=512)
    base_n_layer: int = field(default=4)
    base_num_attention_heads: int = field(default=8)
    
    n_embd: int = field(
        default=512
    )

    max_seq_len: int = field(
        default=512
    )
    activation_cminus: float = field(
        default=-1.,
        metadata={
            "help": (
                "The negative slope for leaky relu will be 1 + activation_cminus / sqrt(width)"
            )
        },
    )
    
    base_attn_mix: float = field(
        default=1.
    )

    n_layer: int = field(
        default=4
    )
    tau0: float = field(
        default=1.
    )
    n_head: int = field(
        default=8
    )
    
    depth_alpha: float = field(
        default=0 # 0 is the default and depth-dependend scaling of residual branches is applied
    )
    
    skip_scaling: float = field(
        default=0.5 
    )

    vocab_size: int = field(
        default=50257
    )
    llama3: Optional[bool] = field(
        default=False, # Temporary command line argument
    )
        
    task: Optional[str] = field(
        default="wikitext",
        metadata={
            "help": (
                "Name of task, e.g. modadd"
            )
        },
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to use auth token.",
            )
        },
    )
    cache_dir: str = field(
        default="/tmp",
        metadata={
            "help": (
                "Huggingface cache directory."
            )
        },
    )
    domains_and_proportions_train: str = field(
        default="{'dclm-0-99-complete': 1.0}",
        metadata={"help": "Domain and proportions for the streaming dataset"}
    )
    domains_and_proportions_val: str = field(
        default="{'refinedweb-172b-len8k': 1.0}",
        metadata={"help": "Domain and proportions for the streaming dataset"}
    )
    streaming_train_root: str = field(
        default="/scratch/gpfs/PLI/conditional_pretraining/packed",
        metadata={"help": "The root directory of the streaming training dataset."}
    )
    streaming_val_root: str = field(
        default="/scratch/gpfs/PLI/conditional_pretraining/packed",
        metadata={"help": "The root directory of the streaming validation dataset."}
    )

if __name__=="__main__":
    from transformers import HfArgumentParser
    parser = HfArgumentParser(TrainingArguments)
    parser.print_help()
