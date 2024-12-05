import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# define custom metrics (f1)
from sklearn.metrics import f1_score

dataset_path = "./mbti_1.csv"  # Path to the CSV file
df = pd.read_csv(dataset_path)

# load the tokenizer from gpt and the model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# using GPT-2 as base model and adding a classification head ontop
model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=16)

# model's padding token set to match tokenizer pad token
model.config.pad_token_id = tokenizer.pad_token_id

# for training on available device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# to tokenize each sample
def tokenize_function(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=128)

# [reparing the dataset with MBTI labels
type_mapping = {t: i for i, t in enumerate(df['type'].unique())}  # map each type to a unique integer
df['labels'] = df['type'].apply(lambda x: type_mapping[x])

def preprocess_dataset(df):
    df['input_ids'] = df['posts'].apply(lambda x: tokenize_function(x)['input_ids'])
    df['attention_mask'] = df['posts'].apply(lambda x: tokenize_function(x)['attention_mask'])
    return df[['input_ids', 'attention_mask', 'labels']]

processed_df = preprocess_dataset(df)

# split the data into train and validation sets
train_df, val_df = train_test_split(processed_df, test_size=0.2, random_state=42)

class MBTIDataset(Dataset):
    def __init__(self, df):
        self.input_ids = list(df['input_ids'])
        self.attention_mask = list(df['attention_mask'])
        self.labels = list(df['labels'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = MBTIDataset(train_df)
eval_dataset = MBTIDataset(val_df)

# computing the class weights and normalize them
labels = df['labels'].values
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
normalized_weights = class_weights / np.sum(class_weights)
class_weights = torch.tensor(normalized_weights, dtype=torch.float).to(device)

# custom loss function
def custom_loss(outputs, labels):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    return loss_fn(outputs, labels)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # **kwargs to handle extra arguments
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = custom_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

# WeightedRandomSampler for balanced batches
class_counts = df['labels'].value_counts().to_dict()
sample_weights = [1.0 / class_counts[label] for label in df['labels']]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler)
eval_loader = DataLoader(eval_dataset, batch_size=2)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=200,
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    num_train_epochs=6,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=1,
    gradient_accumulation_steps=4,
    resume_from_checkpoint="./results/checkpoint-5202",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    f1 = f1_score(labels, predictions, average='weighted')  # Weighted F1-score
    return {"f1": f1}

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Early stopping
)

print("Starting fine tune")
trainer.train()
print("Completed")
