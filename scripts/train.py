import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse
import numpy as np
from datasets import load_metric, Dataset
import evaluate
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, TaskType


def main(args):

    mlb = MultiLabelBinarizer()
    device = "cuda"
    data = pd.read_json("../data/datafinal.json")
    mlb.fit(data["CodeList"])

    train_data = data[data["ContainsCode"].apply(lambda x: isinstance(x, bool))].copy()

    bad_rows = [54, 150, 162, 600, 712, 1599, 1603, 1609, 1611, 1636, 1640, 1718, 1870, 1876, 1879, 1880, 160, 716, 1716, 442, 436, 718]
    train_data = train_data.drop(bad_rows)


    # Salesforce/codet5-base
    # Salesforce/codet5p-220m
    # Salesforce/codegen2-1B
    # Salesforce/codet5p-770m
    # Salesforce/codegen-350M-multi

    
    metric = evaluate.load("google_bleu")
    batch_size = 4
    experiment_name = args.model_name.split("/")[-1] + args.exp_name
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    for t in mlb.classes_:
        if t != tokenizer.decode(tokenizer(t)["input_ids"], skip_special_tokens=True):
            print("tokenizer not supported")
            exit()

    train_data = train_data.sample(frac=1).reset_index(drop=True)
    train_data["CodeList"] = train_data["CodeList"].apply(
        lambda x: x.replace('\\n', '\n').replace('\\t', '\t')
    )

    # train_data["Text"].apply(lambda x: len(x.split())).max()
    # test_data["Text"].apply(lambda x: len(x.split())).max()

    valid_data = train_data.iloc[1200:]
    train_data = train_data.iloc[:1200]

    experiment_train_dataset = Dataset.from_pandas(train_data[["Text"]].reset_index())
    experiment_valid_dataset = Dataset.from_pandas(valid_data[["Text"]].reset_index())

    # Load datasets
    train_dataset = Dataset.from_pandas(train_data[["Text", "CodeList"]].reset_index())
    valid_dataset = Dataset.from_pandas(valid_data[["Text", "CodeList"]].reset_index())


    def preprocess_function_actual(examples):
        inputs = examples["Text"]
        targets = examples["CodeList"]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(text_target=targets)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_function_experiment(examples):
        inputs = examples["Text"]
        targets = examples["Text"]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(text_target=targets)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized_datasets = train_dataset.map(preprocess_function_actual, batched=True)
    valid_tokenized_datasets = valid_dataset.map(preprocess_function_actual, batched=True)

    experiment_train_tokenized_datasets = experiment_train_dataset.map(preprocess_function_experiment, batched=True)
    experiment_valid_tokenized_datasets = experiment_valid_dataset.map(preprocess_function_experiment, batched=True)

    print(f"Training model {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model = model.to(device)
    model.generation_config.max_length = 512

    def print_number_of_trainable_model_parameters(model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


    print(print_number_of_trainable_model_parameters(model))


    # lora_config = LoraConfig(
    #     r=768, # Rank
    #     lora_alpha=1536,
    #     target_modules=["q", "v"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
    # )

    # model = get_peft_model(model, lora_config)
    # print(print_number_of_trainable_model_parameters(model))

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
            
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)    
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)

        result = {"bleu": result["google_bleu"]}
        
        s1 = []
        s2 = []
        
        for x in decoded_preds:
            s1.append(x.replace('\n', '\\n').replace('\t', '\\t'))
        
        for x in decoded_labels:
            s2.append(x[0].replace('\n', '\\n').replace('\t', '\\t'))
        
        t1 = mlb.transform(s1)
        t2 = mlb.transform(s2)
        
        result["mlb_score"] = accuracy_score(t1, t2)
        
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    args = Seq2SeqTrainingArguments(
        f"{experiment_name}",
        evaluation_strategy = "steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        load_best_model_at_end=True,
        eval_steps=300,
        save_steps=300,
        fp16=True,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    ## First Training

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=experiment_train_tokenized_datasets,
        eval_dataset=experiment_valid_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    ## Second Training

    args = Seq2SeqTrainingArguments(
        f"{experiment_name}",
        evaluation_strategy = "steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=15,
        predict_with_generate=True,
        load_best_model_at_end=True,
        eval_steps=300,
        save_steps=300,
        fp16=True,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=valid_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained(f"../{experiment_name}/final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="new_exp"
    )
    args = parser.parse_args()
    main(args)