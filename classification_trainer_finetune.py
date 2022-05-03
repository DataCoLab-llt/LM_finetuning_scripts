import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments, Trainer,
                          DataCollatorForTokenClassification)
from datasets import Dataset, DatasetDict, load_metric
from datasets.utils import disable_progress_bar
disable_progress_bar()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("================================= VM INFO =================================")
print(f"device name : {device}")
print(f"Number of GPU: {torch.cuda.device_count()}")
print(f"Number of CPUs: {os.cpu_count()}")
print(f"GPU type: {torch.cuda.get_device_name(0)}")
print("===========================================================================")

class Config:
    """
    Defining training parameter that use later on for script
    """
    model_id = "allenai/biomed_roberta_base" ## model name on HF hub
    output_dir = "model_output/biomed_roberta_base" ## out_put dir for checkpoints to be saved
    text_col = "text" ## text column for classification
    label_col = "label" ## labels columns to predict
    per_device_train_batch_size =2
    per_device_eval_batch_size=2
    num_train_epochs=1
    num_lable = 2
    data_dir = "path/to/data"
    cols_toRemove_after_tokenization = ["col1", "col2"] ## columns to drop after tokenization
    cpu_num = os.cpu_count() ## for num_proc of tokenization
    save_strategy="no" ## possible save_strategy : no, steps, epoch
    push_to_hub=False ## if True the model push to HF hub
    organization=None
    hub_auth_token=None


tokenizer = AutoTokenizer.from_pretrained(Config.model_id)
model = AutoModelForSequenceClassification.from_pretrained(Config.model_id,
                                                           num_labels=Config.num_lable)

# read data
df = pd.read_csv(Config.data_dir,sep="\t")
df[Config.label_col] = df[Config.label_col].astype(int)
df = df[:100] ## FOR SAMPLE DELETE THIS for training #############################################

#convert pandas dataframe to HF dataset and create collator
dataset = Dataset.from_pandas(df)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)



# separate our data into train, validation and test
train_df = dataset.filter(lambda example: example["split"] == "train")
valid_df = dataset.filter(lambda example: example["split"] == "validation")
test_df = dataset.filter(lambda example: example["split"] == "test")

# tokenize
train_tokenized = train_df.map(lambda e: tokenizer(e['sentence_text_plus'], 
                                          truncation=True, 
                                          padding=True,
                                          max_length=128),
                                          batched=True,
                                          batch_size=16,
                                          num_proc=Config.cpu_num)

valid_tokenized = valid_df.map(lambda e: tokenizer(e['sentence_text_plus'], 
                                          truncation=True, 
                                          padding=True,
                                          max_length=128),
                                          batched=True,
                                          batch_size=16,
                                          num_proc=Config.cpu_num)

test_tokenized = test_df.map(lambda e: tokenizer(e['sentence_text_plus'], 
                                          truncation=True, 
                                          padding=True,
                                          max_length=128),
                                          batched=True,
                                          batch_size=16,
                                          num_proc=Config.cpu_num)

train_tokenized.set_format(type='torch')
valid_tokenized.set_format(type='torch')
test_tokenized.set_format(type='torch')

# remove extra columns
train_tokenized = train_tokenized.remove_columns(Config.cols_toRemove_after_tokenization)
valid_tokenized = valid_tokenized.remove_columns(Config.cols_toRemove_after_tokenization)
test_tokenized = test_tokenized.remove_columns(Config.cols_toRemove_after_tokenization)

## loading metrics to use in training loop
metric = load_metric("accuracy")

## define this function to use in trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if Config.push_to_hub:

    training_args = TrainingArguments(
        output_dir=Config.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        do_eval=True,
        per_device_train_batch_size=Config.per_device_train_batch_size,
        per_device_eval_batch_size=Config.per_device_eval_batch_size,
        num_train_epochs=Config.num_train_epochs,
        fp16=True,
        report_to="tensorboard",
        save_strategy=Config.save_strategy,
        dataloader_drop_last=True,
        push_to_hub=Config.push_to_hub,
        hub_model_id=f"{Config.organization}/finetuned_{Config.model_id.split('/')[-1]}",
        hub_token=Config.hub_auth_token
    )




    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
        
    )


    trainer.train()


    trainer.push_to_hub()
    tokenizer.push_to_hub(f"finetuned_{Config.model_id.split('/')[-1]}",
                        organization=f"{Config.organization}",
                        use_auth_token=Config.hub_auth_token)
    
    
else:
    
    training_args = TrainingArguments(
        output_dir=Config.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        do_eval=True,
        per_device_train_batch_size=Config.per_device_train_batch_size,
        per_device_eval_batch_size=Config.per_device_eval_batch_size,
        num_train_epochs=Config.num_train_epochs,
        fp16=True,
        report_to="tensorboard",
        save_strategy=Config.save_strategy,
        dataloader_drop_last=True
    )



    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
    )


    trainer.train()