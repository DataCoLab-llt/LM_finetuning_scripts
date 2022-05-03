import torch
import os
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          AutoModelForSeq2SeqLM,
                          TrainingArguments, Trainer,
                          PegasusForConditionalGeneration, PegasusTokenizer,
                          DataCollatorForTokenClassification)
from datasets import Dataset, DatasetDict, load_metric, load_dataset
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
    output_dir="./out"
    model_id="sshleifer/distill-pegasus-cnn-16-4"
    num_train_epochs=1
    per_device_train_batch_size =1
    per_device_eval_batch_size=1
    save_strategy="no"
    fp16=True
    report_to ='tensorboard'
    push_to_hub=False
    organization=None
    hub_auth_token=None
    
model = AutoModelForSeq2SeqLM.from_pretrained(Config.model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(Config.model_id)

########################   Dataset class and tokenizer function #####################################
class PegasusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels['input_ids'])  
    

def tokenize_data(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True)
    decodings = tokenizer(labels, truncation=True, padding=True)
    dataset_tokenized = PegasusDataset(encodings, decodings)
    return dataset_tokenized
      

################################################################################################

paws_data = load_dataset('paws', 'labeled_final')


# paraphrasing input
train_texts = paws_data['train']['sentence1'][:10]
train_labels= paws_data['train']['sentence2'][:10]

val_texts = paws_data['validation']['sentence1'][:10]
val_labels= paws_data['validation']['sentence2'][:10]


train_dataset = tokenize_data(train_texts, train_labels)
val_dataset = tokenize_data(val_texts, val_labels)
       


if Config.push_to_hub: 
    
    training_args = TrainingArguments(
        output_dir=Config.output_dir, 
        num_train_epochs=Config.num_train_epochs,         
        per_device_train_batch_size=Config.per_device_train_batch_size, 
        per_device_eval_batch_size=Config.per_device_eval_batch_size,   
        report_to = Config.report_to,
        save_strategy=Config.save_strategy,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        do_eval=True,
        fp16=Config.fp16,
        dataloader_drop_last=True,
        push_to_hub=Config.push_to_hub,
        hub_model_id=f"{Config.organization}/finetuned_{Config.model_id.split('/')[-1]}",
        hub_token=Config.hub_auth_token
    )



    trainer = Trainer(
        model=model,                         
        args=training_args,                 
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer

    )

    trainer.train()
    
    trainer.push_to_hub()
    tokenizer.push_to_hub(f"finetuned_summarization_{Config.model_id.split('/')[-1]}",
                        organization=Config.organization,
                        use_auth_token=Config.hub_auth_token)
    
    
    
    
else:
    
    training_args = TrainingArguments(
        output_dir=Config.output_dir, 
        num_train_epochs=Config.num_train_epochs,         
        per_device_train_batch_size=Config.per_device_train_batch_size, 
        per_device_eval_batch_size=Config.per_device_eval_batch_size,   
        report_to = Config.report_to,
        save_strategy=Config.save_strategy,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        do_eval=True,
        fp16=Config.fp16,
        dataloader_drop_last=True
    )



    trainer = Trainer(
        model=model,                         
        args=training_args,                 
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer

    )

    trainer.train()