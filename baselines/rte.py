import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import random
import numpy as np
import time
import pandas as pd

BATCH_SIZE = 32
EPOCHS = 3
model_name = "bert-base-uncased"
train_path = './RTE/train.tsv'
test_path = './RTE/test.tsv'

train_df = pd.read_csv(train_path, sep='\t')
test_df = pd.read_csv(test_path, sep='\t')

train_path_new = './RTE/train.csv'
test_path_new = './RTE/test.csv'

train_df.to_csv(train_path_new, index = False)
test_df.to_csv(test_path_new, index = False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset('csv', data_files={'train': train_path_new})
test_dataset = load_dataset('csv', data_files={'test': test_path_new})

tokenizer = BertTokenizer.from_pretrained(model_name)


def preprocess_data(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], padding='max_length',
                     truncation=True, max_length=256)


def label_setter(examples):
    return {'labels': torch.tensor(1 if examples['label']=="not_entailment" else 0, dtype=torch.long)}


train_dataset = dataset['train'].map(preprocess_data, batched=True)
test_dataset = test_dataset['test'].map(preprocess_data, batched=True)
train_dataset = train_dataset.map(label_setter)
test_dataset = test_dataset.map(label_setter)

train_dataset = train_dataset.remove_columns(['premise','hypothesis','label'])
test_dataset = test_dataset.remove_columns(['premise','hypothesis','label'])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

print(test_dataset[2:4])
print("Preprocessing done")

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=BATCH_SIZE)

test_dataloader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=BATCH_SIZE)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, output_attentions=False,
                                                      output_hidden_states=False)

model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


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
        loss, logits = model_output.loss, model_output.logits
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = time.time() - t0
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
    print("Running Validation...")

    model.eval()
    total_eval_loss = 0
    correct = 0
    total = 0
    for batch in test_dataloader:
        input_ids, token_type_ids, attention_mask, labels = batch['input_ids'].to(device), batch['token_type_ids'].to(
            device), batch['attention_mask'].to(device), batch['labels'].to(device)

        with torch.no_grad():
            model_output_val = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                   labels=labels)
        loss, logits = model_output_val.loss, model_output_val.logits

        total_eval_loss += loss.item()

        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f'Epoch: {epoch}, Val accuracy: {val_acc}, Val loss: {total_eval_loss/len(test_dataloader)}')
