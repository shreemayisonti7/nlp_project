import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QQPDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=75):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        inputs = self.tokenizer(item['question1'], item['question2'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        labels = torch.tensor(item['label'])
        return inputs, labels

# Load the QQP dataset from the GLUE benchmark
qqp_dataset = load_dataset("glue", "qqp")

# Access the train, validation, and test splits
train_dataset = pd.DataFrame(qqp_dataset["train"])
validation_dataset = pd.DataFrame(qqp_dataset["validation"])
test_dataset = pd.DataFrame(qqp_dataset["test"])

# Load the tokenizer and model
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

# Define hyperparameters
batch_size = 128
num_epochs = 3
learning_rate = 5e-5

# Create PyTorch datasets and dataloaders
train_dataset = QQPDataset(train_dataset, tokenizer)
val_dataset = QQPDataset(validation_dataset, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Set up the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = batch
       
        inputs = {key: value.squeeze(1).to(device) for key, value in inputs.items()}
      
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels,return_dict=True)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validation
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = batch
            inputs = {key: value.squeeze(1).to(device) for key, value in inputs.items()}
            labels = labels.to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}")

# Save the model
model.save_pretrained("saved_model_epoch2")

# Inference example
def check_similarity(question1, question2):
    inputs = tokenizer(question1, question2, truncation=True, padding='max_length', max_length=75, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    logits = model(**inputs).logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return predicted_class_id

# Example usage
result = check_similarity("Why are people so obsessed with cricket?", "Why do people like cricket?")
print(result)

