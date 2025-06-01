import transformers

from llm_finetune.arguments import (
    ModelArguments,
    DataArguments,
    TRL_GRPOTrainingArguments,
)
from llm_finetune.dataset import make_reasoning_data_module_trl
from llm_finetune.grpo import (
    xmlcount_reward_func,
    soft_format_reward_func,
    correctness_reward_func,
    reasoning_length_reward
)

from trl import GRPOTrainer
import datetime
import os



def train():
    
    # HFArgumentParser is a custom argument parser that can handle nested dataclasses
    # https://huggingface.co/docs/transformers/en/internal/trainer_utils
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TRL_GRPOTrainingArguments)
    )
    
    # Returns Tuple of dataclasses
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args, 'training_args')

    if '{}' in training_args.output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        training_args.output_dir = training_args.output_dir.format(timestamp) #Change output file name

    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Get Model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # Get Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        chatml_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% if loop.last and add_generation_prompt %}"
            "{{'<|im_start|>assistant\n'}}"
            "{% endif %}"
            "{% endfor %}"
        )
        tokenizer.chat_template = chatml_template

    # Format dataset from csv to Dataset
    data_module = make_reasoning_data_module_trl(
        data_args=data_args,
    )
    
    # Set up training arguments
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            reasoning_length_reward,
            correctness_reward_func,
        ],
        **data_module
    )
    
    # Using GRPO Trainer 
    trainer.train(training_args.checkpoint)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
