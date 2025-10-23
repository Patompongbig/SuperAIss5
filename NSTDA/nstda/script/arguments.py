from dataclasses import dataclass, field
from typing import Optional, Union

from trl import SFTConfig

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/project/lt200246-mmacma/Big_seq2seq/model/Qwen/Qwen2.5-3B")
    save_path: Optional[str] = field(default="/project/lt200246-mmacma/Big_seq2seq/model/Qwen/Qwen2.5-3B")

@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )

@dataclass
class TRL_SFTTrainingArguments(SFTConfig):
    output_dir: Optional[str] = field(default='/project/lt200246-mmacma/Big_seq2seq/sft_llm/trained_model/qwen2.5/')
    save_strategy: str = "no"
    max_seq_length: int = 2048
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = field(default=4)
    gradient_checkpointing: bool = True
    optim: str="adamw_torch_fused"
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    max_grad_norm: float = 0.2
    lr_scheduler_type: str = "cosine"
    seed: int = 3407
    report_to: str = "wandb"

    do_train: bool = True
    do_eval: bool = True
    eval_strategy: str = "epoch"
    logging_strategy: str = "epoch"
    push_to_hub: bool = False


# --logging_steps 5 \
# logging_steps: int = 10