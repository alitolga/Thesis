# This is a script I added, in case I need to use a manual training loop
#
# This is not the script I am currently using for my thesis,
# I am training the models using the code in training_models_causalLM_new.py
#

from peft import LoraConfig, TaskType, get_peft_model 
from transformers import BertTokenizer, AutoModelForCausalLM, AdamW
from datasets import load_dataset

import numpy as np
import torch

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

peft_config_bert = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8,
    lora_alpha=32, 
    lora_dropout=0.1)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

bert_model_unused = AutoModelForCausalLM.from_pretrained("bert-base-uncased")

bert_model_peft = get_peft_model(bert_model_unused, peft_config_bert)
bert_model_peft.print_trainable_parameters()

# Specify the dataset name
dataset_name = "bookcorpus"

# Specify the path to save or load the dataset
save_path = "./data"

# Load the dataset, use the cache if available
dataset = load_dataset(dataset_name, cache_dir=save_path)

bookcorpus_dataset = dataset["train"]["text"][:40000]

print(len(bookcorpus_dataset))
print(bookcorpus_dataset[0])

tokenized_inputs = bert_tokenizer(bookcorpus_dataset, return_tensors='pt', truncation=True, padding=True)
print(type(tokenized_inputs))
print(tokenized_inputs['input_ids'].shape)

class CustomDataset(Dataset):
    def __init__(self, batch_encoding):
        self.batch_encoding = batch_encoding

    def __len__(self):
        return len(self.batch_encoding.input_ids)

    def __getitem__(self, index):
        # Extract tensors from BatchEncoding
        input_ids = self.batch_encoding.input_ids[index]
        attention_mask = self.batch_encoding.attention_mask[index]
        token_type_ids = self.batch_encoding.token_type_ids[index]

        # Convert to dictionary
        inputs_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

        return inputs_dict
    
# Assuming batch_encoding is your BatchEncoding object
training_dataset = CustomDataset(tokenized_inputs)

bert_model_peft.to('cuda')

num_train_epochs = 500
per_device_train_batch_size = 32
learning_rate = 1e-3

dataloader = DataLoader(training_dataset, batch_size=per_device_train_batch_size, shuffle=True)

# Define the optimizer
optimizer = AdamW(bert_model_peft.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_train_epochs):
    bert_model_peft.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_train_epochs}"):
        inputs = {key: value.to('cuda') for key, value in batch.items()}

        # Forward pass
        outputs = bert_model_peft(**inputs, labels=inputs["input_ids"])
        # outputs = bert_model_peft(**inputs)

        # Calculate loss
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_train_epochs}, Average Loss: {average_loss}")

bert_model_peft.save_pretrained("./models/bert-lora")
