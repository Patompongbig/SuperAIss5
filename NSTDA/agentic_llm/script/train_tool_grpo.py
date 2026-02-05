import transformers

from grpo_setting import (
    get_agnews_questions_for_tools,
    think_quality_reward,
    repetition_penalty_reward,
    lexical_and_semantic_reward_function,
    oov_error_reward_function,
)

from arguments import (
    ModelArguments,
    DataArguments,
    TRL_GRPOTrainingArguments
)

from tools.tool_rag import (
    retrieve_similar_thai_gloss_sentences, 
    get_oov_gloss_tokens,
)

from datasets import load_dataset
from trl import GRPOTrainer, clone_chat_template
from trl.chat_template_utils import add_response_schema
import torch
import datetime
import os
import time
import evaluate

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TRL_GRPOTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args, 'training_args')

    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        cache_dir=None,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path
    )

    # tokenizer = add_response_schema(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = load_dataset(data_args.train_data, cache_dir=None)
    data_module = get_agnews_questions_for_tools(dataset)
    
    print(data_module)
    print(data_module["grpo_train"][0])

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        reward_funcs=[lexical_and_semantic_reward_function,
                    oov_error_reward_function,
                    think_quality_reward,
                    repetition_penalty_reward
                    ],
        train_dataset=data_module["grpo_train"],
        eval_dataset=data_module["grpo_eval"],
        tools=[
            retrieve_similar_thai_gloss_sentences,
            get_oov_gloss_tokens
        ]
    )

    trainer.train()
    trainer.save_model()
    time.sleep(300)
    print("Completed Training GRPO Model")

if __name__ == "__main__":
    train()
