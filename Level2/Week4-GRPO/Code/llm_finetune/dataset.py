from datasets import load_from_disk
from .grpo import process_agnews_dataset_for_reasoning
from .grpo import SYSTEM_PROMPT_REASONING


# Get the dataset by loading and mapping to the system and user prompts
def make_reasoning_data_module_trl(data_args):
    """(New Method) Make dataset for TRL supervised fine-tuning."""

    # Load the dataset from disk and process to prompt format
    train_dataset = process_agnews_dataset_for_reasoning(load_from_disk(data_args.train_data_path))
    eval_dataset = None
    
    # If an evaluation dataset path is provided, load and process it as well
    if data_args.eval_data_path:
        eval_dataset = process_agnews_dataset_for_reasoning(load_from_disk(data_args.eval_data_path))
    
    # Return as dictionary with train and eval datasets
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )



# Direct using the response as the answer
def get_agnews_questions_for_sft(dataset):
  dataset = dataset.map(lambda x: { # type: ignore
      'prompt': [
          {'role': 'system', 'content': SYSTEM_PROMPT_REASONING},
          {'role': 'user', 'content': x['input_train']}
      ],
      'completion': [
          {'role': 'assistant', 'content': x['answer']}
      ],
  }) # type: ignore
  return dataset # type: ignore


def make_supervised_data_module_trl(data_args):
    """(New Method) Make dataset for TRL supervised fine-tuning."""

    train_dataset = get_agnews_questions_for_sft(load_from_disk(data_args.train_data_path))
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = get_agnews_questions_for_sft(load_from_disk(data_args.eval_data_path))
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )