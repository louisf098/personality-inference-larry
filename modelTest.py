from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

model = AutoModelForSequenceClassification.from_pretrained('./results/checkpoint-5202')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

type_mapping = {
    'INFJ': 0,
    'ENTP': 1,
    'INTP': 2,
    'INTJ': 3,
    'ENTJ': 4,
    'ENFJ': 5,
    'INFP': 6,
    'ENFP': 7,
    'ISFP': 8,
    'ISTP': 9,
    'ISFJ': 10,
    'ISTJ': 11,
    'ESTP': 12,
    'ESFP': 13,
    'ESTJ': 14,
    'ESFJ': 15
}

reverse_type_mapping = {v: k for k, v in type_mapping.items()}

# Sample text to classify
text = "What? High school was great. I was either stoned stupid or asleep, woke up just long enough to bullshit the tests/exams."

# Tokenize the input text for testing
inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")

# Make prediction
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient computation
    outputs = model(**inputs)

# Get logits and convert to probabilities
logits = outputs.logits
probs = F.softmax(logits, dim=-1)

# Print logits and probabilities for tesintg and debugging
print("Logits:", logits)
print("Probabilities:", probs)

# Get the predicted class index and map it to MBTI type
predicted_class = torch.argmax(probs, dim=-1).item()
predicted_type = reverse_type_mapping[predicted_class]

print(f"Predicted MBTI type: {predicted_type}")
