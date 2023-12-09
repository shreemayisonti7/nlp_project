from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from datasets import load_metric
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import random
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def preprocess_data(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)


def label_setter(examples):
    if examples['label'] == 'Negative':
        return {'labels': torch.Tensor([1, 0, 0])}
    elif examples['label'] == 'Positive':
        return {'labels': torch.Tensor([0, 1, 0])}
    else:
        return {'labels': torch.Tensor([0, 0, 1])}


model_name = "bert-base-uncased"
BATCH_SIZE = 32
tokenizer = BertTokenizer.from_pretrained(model_name)

dataset = load_dataset("json", data_files={"train": f"./../data/train.jsonl",
                                           "validation": f"./../data/validation.jsonl"})

train_dataset = dataset['train'].map(preprocess_data, batched=True)
val_dataset = dataset['validation'].map(preprocess_data, batched=True)

train_dataset = train_dataset.map(label_setter)
val_dataset = val_dataset.map(label_setter)

remove_keys = [col for col in dataset["train"].column_names if col not in ["input_ids",
                                                                           "token_type_ids",
                                                                           "attention_mask", "labels"]]

train_dataset = train_dataset.remove_columns(remove_keys)
val_dataset = val_dataset.remove_columns(remove_keys)

train_dataset.set_format("torch")
val_dataset.set_format("torch")

print(val_dataset[0])

print("Preprocessing done")

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=BATCH_SIZE)

validation_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=BATCH_SIZE)

model_name = "bert-base-uncased"
num_labels = 3
learning_rate = 2e-5
EPOCHS = 5

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,
                                                      output_attentions=False, output_hidden_states=False)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

start_time = time.time()
for epoch in range(EPOCHS):
    t0 = time.time()
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        input_ids, token_type_ids, attention_mask, labels = batch['input_ids'].to(device), batch['token_type_ids'].to(
            device), batch['attention_mask'].to(device), batch['labels'].to(device)
        model.zero_grad()
        model_output = model(input_ids,
                             token_type_ids=token_type_ids,
                             attention_mask=attention_mask,
                             labels=labels)
        loss = model_output.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = time.time() - t0
    print("Average training loss: {0:.2f}".format(avg_train_loss))
    print("Training epoch took: {:}".format(training_time))
    print("Running Validation...")

    model.eval()
    total_eval_loss = 0
    correct = 0
    total = 0
    for batch in validation_dataloader:
        input_ids, token_type_ids, attention_mask, labels = batch['input_ids'
                                                            ].to(device), batch['token_type_ids'].to(device), batch[
                                                                'attention_mask'
                                                            ].to(device), batch['labels'].to(device)

        with torch.no_grad():
            model_output_val = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                     labels=labels)
        loss, logits = model_output_val.loss, model_output_val.logits

        total_eval_loss += loss.item()

        _, predicted = torch.max(logits, 1)
        _, labels = torch.max(labels, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f'Epoch: {epoch}, Val accuracy: {val_acc}, Val loss: {total_eval_loss / len(validation_dataloader)}')

end_time = time.time()

print(f'Total time taken for fine tuning {end_time - start_time}')
