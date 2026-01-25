from dataclasses import dataclass, field
from typing import Optional, Union

from trl import GRPOConfig

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/project/lt-user/pretrained_model/Qwen/Qwen3-14B")
    save_path: Optional[str] = field(default="/project/lt-user/pretrained_model/Qwen/Qwen3-14B")

@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )

@dataclass
class TRL_GRPOTrainingArguments(GRPOConfig):
    output_dir: Optional[str] = field(default='/project/lt-user/agent_grpo/qwen3_14B_tool')
    save_strategy: str = "no"

    max_completion_length: int = 256
    num_generations: int = 8

    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = field(default=4)
    gradient_checkpointing: bool = True

    optim: str="adamw_torch_fused"
    learning_rate: float = 3e-6
    warmup_steps: int = 5
    max_grad_norm: float = 0.2
    lr_scheduler_type: str = "cosine"
    seed: int = 3407
    report_to: str = "wandb"

    do_train: bool = True
    do_eval: bool = True
    eval_strategy: str = "steps"
    eval_steps = 100
    logging_strategy: str = "steps"
    logging_steps = 10
    push_to_hub: bool = False

    use_vllm: bool = True
    vllm_mode: str = "server"

    use_bias_correction_kl: bool = True
    