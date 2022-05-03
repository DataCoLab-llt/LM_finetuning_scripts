
from datasets import load_dataset, load_metric
from datasets.utils import disable_progress_bar
import numpy as np
import os
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          TrainingArguments,
                          Trainer)
disable_progress_bar()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("================================= VM INFO =================================")
print(f"device name : {device}")
print(f"Number of GPU: {torch.cuda.device_count()}")
print(f"Number of CPUs: {os.cpu_count()}")
print(f"GPU type: {torch.cuda.get_device_name(0)}")
print("===========================================================================")


###################################### Functions #####################################################

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True,max_length=128)



def group_texts(examples, block_size=128):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

######################################################################################################

class Config:
    """
    Defining training parameter that use later on for script
    """
    train_data_id = "RohanAiLab/persian_blog_V2"
    model_id = "flax-community/gpt2-medium-persian"
    output_dir = "out"
    num_train_epochs = 1
    per_device_train_batch_size = 16
    per_device_eval_batch_size=8
    overwrite_output_dir = True
    cpu_num = os.cpu_count() ## for num_proc of tokenization
    save_strategy="no" ## possible save_strategy : no, steps, epoch
    push_to_hub=False ## if True the model push to HF hub
    push_to_hub_token=None
    push_to_hub_model_id=None
    report_to="tensorboard"
    fp16=True
    
    
    

dataset = load_dataset(Config.train_data_id, revision="dev") ### DEV

tokenizer = AutoTokenizer.from_pretrained(Config.model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(Config.model_id)

tokenized_datasets = dataset.map(tokenize_function,
                                batched=True,
                                batch_size=128,
                                num_proc=os.cpu_count(),
                                remove_columns=["text"])



lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=128,
    num_proc=os.cpu_count(),
)

if Config.push_to_hub:
    training_args = TrainingArguments(output_dir=Config.output_dir,
                                    num_train_epochs=Config.num_train_epochs,
                                    overwrite_output_dir=Config.overwrite_output_dir,
                                    do_eval=True,
                                    per_device_train_batch_size=Config.per_device_train_batch_size,
                                    push_to_hub=Config.push_to_hub,
                                    push_to_hub_model_id=Config.push_to_hub_model_id,
                                    push_to_hub_token=Config.push_to_hub_token,
                                    save_strategy=Config.save_strategy,
                                    report_to=Config.report_to,
                                    fp16=Config.fp16
                                    )




    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],

    )


    trainer.train()
    
    trainer.push_to_hub(commit_message="PUSH model")
    
    
else:
    
    training_args = TrainingArguments(output_dir=Config.output_dir,
                                    num_train_epochs=Config.num_train_epochs,
                                    overwrite_output_dir=Config.overwrite_output_dir,
                                    do_eval=True,
                                    per_device_train_batch_size=Config.per_device_train_batch_size,
                                    save_strategy=Config.save_strategy,
                                    report_to=Config.report_to,
                                    fp16=Config.fp16
                                    )




    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],

    )
    
    trainer.train()

        
        
        
