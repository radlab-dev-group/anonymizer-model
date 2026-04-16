import os
import json
import wandb

import numpy as np
from datasets import Dataset
from datetime import datetime
from seqeval.metrics import classification_report

from transformers import (
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

from data_processor import NERDataProcessor


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index -100 and convert to labels
    true_predictions = [
        [label_map[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_map[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_predictions, output_dict=True)

    # Wyciągamy średnie metryki (macro avg), aby wandb mógł je wykreślić
    return {
        "f1": report["macro avg"]["f1-score"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
    }


def main(config_path: str, data_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(config["output_dir"], now_str)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Setup W&B
    run_name = f"run_{now_str}"
    wandb.init(project=config["wb_project"], name=run_name, config=config)

    # 2. Data Processing
    processor = NERDataProcessor(config["model_name"], config["max_length"])
    raw_data = processor.load_jsonl(data_path)
    label2id, id2label = processor.create_label_mappings(raw_data)

    # Save mappings
    with open(os.path.join(out_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)
    with open(os.path.join(out_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f)

    # Prepare Dataset
    # Split data (80/20)
    split = int(len(raw_data) * 0.8)
    train_raw = raw_data[:split]
    eval_raw = raw_data[split:]

    def prepare_dataset(data_list):
        texts = [item["text"] for item in data_list]
        labels = [item["labels"] for item in data_list]
        encoded = processor.tokenize_and_align_labels(
            {"text": texts, "labels": labels}, label2id
        )
        return Dataset.from_dict(encoded)

    train_dataset = prepare_dataset(train_raw)
    eval_dataset = prepare_dataset(eval_raw)

    # 3. Model Initialization
    model = AutoModelForTokenClassification.from_pretrained(
        config["model_name"],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # Global map for metrics
    global label_map
    label_map = id2label

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(processor.tokenizer),
    )

    trainer.train()

    # Save final model
    final_model_dir = os.path.join(out_dir, "final_model")
    trainer.save_model(final_model_dir)
    processor.tokenizer.save_pretrained(final_model_dir)

    wandb.finish()


if __name__ == "__main__":
    # Assuming data.jsonl exists with the provided samples
    main(
        "config/training/kpwr-ner-config.json",
        "dataset/kpwr/converted/generalised/kpwr-ner-general-whole.jsonl",
    )
