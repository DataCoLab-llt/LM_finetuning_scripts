import os
import torch
from datasets import load_dataset
from transformers import (Seq2SeqTrainingArguments,
                         Seq2SeqTrainer,
                         AutoTokenizer,
                         AutoModelForSeq2SeqLM,
                         DataCollatorForSeq2Seq)
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
    model_id = "t5-small"
    output_dir = "/home/jupyter/model_output/t5-small"
    text_col = "text"
    summary_col = "summary"
    evaluation_strategy="epoch"
    per_device_train_batch_size=8
    per_device_eval_batch_size=8
    num_train_epochs=1
    report_to="tensorboard"
    push_to_hub=False ## if True the model push to HF hub
    organization=None
    hub_auth_token=None
    fp16=True
    cpu_num=os.cpu_count()
    
    
tokenizer = AutoTokenizer.from_pretrained(Config.model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(Config.model_id)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


# read data
billsum = load_dataset("billsum", split="ca_test")
billsum = billsum.select(range(100)) ################### JUST FOR TESTINT ####################
billsum = billsum.train_test_split(test_size=0.2)

# preprocess data
prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples[Config.text_col]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[Config.summary_col], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# tokenize
tokenized_billsum = billsum.map(preprocess_function, batched=True, num_proc=Config.cpu_num)


if Config.push_to_hub:
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy=Config.evaluation_strategy,
        per_device_train_batch_size=Config.per_device_train_batch_size,
        per_device_eval_batch_size=Config.per_device_eval_batch_size,
        num_train_epochs=Config.num_train_epochs,
        report_to=Config.report_to,
        fp16=Config.fp16,
        push_to_hub=Config.push_to_hub,
        hub_model_id=f"{Config.organization}/finetuned_{Config.model_id.split('/')[-1]}",
        hub_token=Config.hub_auth_token
    )

    # set up trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_billsum["train"],
        eval_dataset=tokenized_billsum["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # training model
    trainer.train()
    
    trainer.push_to_hub()
    tokenizer.push_to_hub(f"finetuned_summarization_{Config.model_id.split('/')[-1]}",
                        organization=Config.organization,
                        use_auth_token=Config.hub_auth_token)
    
else:
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=Config.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy=Config.evaluation_strategy,
        per_device_train_batch_size=Config.per_device_train_batch_size,
        per_device_eval_batch_size=Config.per_device_eval_batch_size,
        num_train_epochs=Config.num_train_epochs,
        report_to=Config.report_to,
        fp16=Config.fp16,
    )

    # set up trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_billsum["train"],
        eval_dataset=tokenized_billsum["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # training model
    trainer.train()