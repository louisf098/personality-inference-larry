import torch
import kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM

# Download latest version
path = kagglehub.dataset_download("datasnaek/mbti-type")

print("Path to dataset files:", path)

# Load the tokenizer for GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load the pre-trained GPT-2 model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

