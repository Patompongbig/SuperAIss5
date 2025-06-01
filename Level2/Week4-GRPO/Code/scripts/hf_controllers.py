import os

# HfAPI : https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api
# snapshot download : https://huggingface.co/docs/huggingface_hub/en/guides/download
from huggingface_hub import snapshot_download, HfApi, upload_folder
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from typing import Optional, List



class HuggingFaceController:
    
    def __init__(self, hf_token : str = None):
        
        self.hf_token = hf_token
        self.hf_api = HfApi(hf_token=self.hf_token) if self.hf_token else HfApi()
        
        
        
        
    def download_model_tokenize(self, model_name: str, output_dir: str):
        """
        Downloads a specific model and its tokenizer from Hugging Face and saves them to the specified directory.

        Args:
            model_name (str): The name of the model on Hugging Face (e.g., 'bert-base-uncased').
            output_dir (str): The local directory where the model and tokenizer will be saved.

        Returns:
            tuple: A tuple containing the loaded tokenizer and model, or None if download fails.
        """
        
        try :
            snapshot_download(
                repo_id=model_name,
                local_dir=output_dir,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
            model = AutoModelForSequenceClassification.from_pretrained(output_dir)
            
            return tokenizer, model
        
        except Exception as e:
            print(f"Error downloading model and tokenizer: {e}")
            return None, None
        
    
    
    
    
    def upload_model_to_huggingface(self, model_path: str, repo_name: str, private: bool = False):
        """
        Uploads a model and its associated files (including tokenizer) from a local directory to Hugging Face.

        Args:
            model_path (str): The local directory containing the model files (e.g., config.json, pytorch_model.bin, tokenizer.json).
            repo_name (str): The name of the repository to create/update on your Hugging Face account (e.g., 'my-awesome-model').
            private (bool, optional): Whether the repository should be private. Defaults to False.

        Returns:
            str: The URL of the uploaded repository on Hugging Face, or None if upload fails.
        """
        if not self.hf_token:
            print("Error: Hugging Face token is required for uploading.")
            return None

        try:
            repo_id = f"{self.hf_api.whoami()['name']}/{repo_name}"
            self.hf_api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            
            for filename in os.listdir(model_path):
                
                # Create a path
                file_path = os.path.join(model_path, filename)
                
                # Upload the file to hugging face
                self.hf_api.upload_file(path_or_fileobj=file_path,
                                        path_in_repo=filename,
                                        repo_id=repo_id,)
            
            repo_url = f"https://huggingface.co/{repo_id}"
            print(f"Model and tokenizer uploaded successfully to: {repo_url}")
            
            return repo_url
        
        except Exception as e:
            print(f"Error uploading model to Hugging Face: {e}")
            return None
        
        
        
        
    def zero_shot_predict(self, model_name: str, candidate_labels: list, text: str, multi_label: bool = False):
        """
        Performs zero-shot classification using a specified Hugging Face model.

        Args:
            model_name (str): The name of the zero-shot classification model on Hugging Face (e.g., 'facebook/bart-large-mnli').
            candidate_labels (list): A list of candidate labels for classification.
            text (str): The text to classify.
            multi_label (bool, optional): Whether to allow multiple labels to be predicted. Defaults to False.

        Returns:
            dict: A dictionary containing the predicted labels and their scores, or None if prediction fails.
        """
        try:
            classifier = pipeline("zero-shot-classification", model=model_name)
            result = classifier(text, 
                                candidate_labels, 
                                multi_label=multi_label)
            
            print(f"Zero-shot prediction result for '{text}': {result}")
            return result
        
        except Exception as e:
            print(f"Error during zero-shot prediction: {e}")
            return None
        
        
if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = f"/project/ai901504-ai0004/500101-Boss/superai-llm-reasoning/SuperAI_LLM_FineTune_2025/interesting_model/{model_name.replace('/', '_')}"
    
    hf_controller = HuggingFaceController(hf_token=hf_token)
    
    tokenizer, model = hf_controller.download_model_tokenize(model_name=model_name, output_dir=output_dir)
    
    if tokenizer and model:
        print("Model and tokenizer downloaded successfully.")