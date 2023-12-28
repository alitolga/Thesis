#!/usr/bin/env python
# coding: utf-8

import torch
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer
from functools import partial
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model 
import math
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DebertaV2ForMaskedLM


# #### Import the bookcorpus dataset

save_path = "./data"

bookcorpus_dataset = load_dataset("bookcorpus", split="train[:50000]", cache_dir=save_path)
bookcorpus_dataset = bookcorpus_dataset.train_test_split(test_size=0.2)

def train(modelname):

    # #### Double checking we are using the GPU on the VSC  
    # Check if CUDA is available and set the device to GPU if it is
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # ### Preprocessing

    # Import the tokenizer

    tokenizer = AutoTokenizer.from_pretrained(modelname)


    transformers.logging.set_verbosity_info()

    # Preprocessing Function 1 - Map the data to the tokenizer function
  
    def preprocess_function(tokenizer, examples):
        return tokenizer([" ".join(x) for x in examples["text"]])

    partial_tokenize_function = partial(preprocess_function, tokenizer)

    tokenized_bookcorpus = bookcorpus_dataset.map(
        partial_tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=bookcorpus_dataset["train"].column_names,
    )

    # tokenized_bookcorpus

    # Tokenizer Function 2 - Divide the dataset into blocks of block size. Drop the remainder if the length of the dataset is not fully divisible to the block size.

    def group_texts(examples):
        block_size = 128

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    lm_dataset = tokenized_bookcorpus.map(group_texts, batched=True, num_proc=4)
    # Import a Data Collator Function for (Causal) LM. This function will ensure that for each token, we have the following token respective to it as it's label/target.

    # tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)


    # #### Import the LoRA library from PEFT. Set it's parameters and load the model optimized using LoRA

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8,
        lora_alpha=32, 
        lora_dropout=0.1)


    # We can see the reduced number of parameters below

    if modelname == "microsoft/deberta-base" or modelname == "microsoft/deberta-v3-base":
        model_without_peft = DebertaV2ForMaskedLM.from_pretrained(modelname)
    else:
        model_without_peft = AutoModelForCausalLM.from_pretrained(modelname)

    model = get_peft_model(model_without_peft, peft_config)

    model.print_trainable_parameters()
    print(next(model.parameters()).device)

    # #### Set the Training Arguments
    batch_size = 32

    training_args = TrainingArguments(
        output_dir=f"mymodels/{modelname}-large-peft",
        evaluation_strategy="epoch",
        # learning_rate=2e-5,
        # weight_decay=0.01,
        learning_rate=5e-3,
        per_device_train_batch_size=batch_size,
        # gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        push_to_hub=True,
        report_to="all",
        logging_dir='./logs',            
        logging_steps=100,
    )


    # If the tokenizer doesn't have a padding token by default, use End of Sequence Token. If it also doesn't have that, then we have to use a Separator or a Classification token...

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.cls_token

    print("Tokenizer PADDING Token is: ")
    print(tokenizer.pad_token)

    # Ensure that we are running the model on Gpu and not on Cpu

    print(next(model.parameters()).device)

    model.to(device)

    print(next(model.parameters()).device)


    # #### Finally create the Trainer class and train the model


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    # ##### Evaluate the model using Cosine Similarity, Pairwise Correlation...
    # Perplexity is just there as a placeholder for now

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


    # Finally push the model to the Huggingface Hub

    # trainer.save_model(f"{modelname}-peft")
    # model.save_pretrained(f"{modelname}-peft-model")
    trainer.push_to_hub()


if __name__ == '__main__':

    modelnames = []
    # modelnames.append("bert-base-uncased")
    # modelnames.append("roberta-base")
    # modelnames.append("google/electra-base-generator")
    # modelnames.append("facebook/bart-base")
    # modelnames.append("gpt2")
    # modelnames.append("microsoft/deberta-base")
    # modelnames.append("microsoft/deberta-v3-base")

    for modelname in modelnames:
        train(modelname)

