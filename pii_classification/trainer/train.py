import os
import json
import torch
import wandb
import random

import numpy as np

from datasets import Dataset
from datetime import datetime
from sklearn.metrics import classification_report

from transformers import (
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

from rdl_ml_utils.handlers.wandb_handler import WanDBHandler

from data_processor import AnnoDataProcessor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _WandbConfig:
    PROJECT_NAME: str = os.getenv("WANDB_PROJECT", "kpwr-ner")
    PROJECT_TAGS: list = (
        os.getenv("WANDB_TAGS", "").split(",") if os.getenv("WANDB_TAGS") else []
    )
    PREFIX_RUN: str = os.getenv("WANDB_PREFIX", "run_")
    BASE_RUN_NAME: str = os.getenv("WANDB_BASE", "ner-model")


class WandbLoggingCallback(TrainerCallback):
    def __init__(self, handler: WanDBHandler):
        self.handler = handler

    def on_log(
        self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs
    ):
        if logs is None:
            return

        is_eval = any(key.startswith("eval_") for key in logs.keys())
        step = int(state.global_step) + 1 if is_eval else int(state.global_step)
        self.handler.log_metrics(logs, step=step)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_map[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_map[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_predictions_flat = [p for seq in true_predictions for p in seq]
    true_labels_flat = [l for seq in true_labels for l in seq]

    report = classification_report(
        true_labels_flat, true_predictions_flat, output_dict=True
    )

    return {
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
    }


def main(config_path: str, data_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(config["output_dir"], now_str)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Data Processing
    processor = AnnoDataProcessor(config["model_name"], config["max_length"])
    raw_data = processor.load_jsonl(data_path)
    random.shuffle(raw_data)

    label2id, id2label = processor.create_label_mappings(raw_data)

    with open(os.path.join(out_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)
    with open(os.path.join(out_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f)

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

    # 2. Model Initialization
    model = AutoModelForTokenClassification.from_pretrained(
        config["model_name"],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    global label_map
    label_map = id2label

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
        seed=seed,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        eval_strategy=config.get("eval_strategy", "steps"),
        logging_strategy=config.get("logging_strategy", "steps"),
        save_strategy=config.get("save_strategy", "steps"),
        eval_steps=config.get("eval_steps", 100),
        logging_steps=config.get("logging_steps", 50),
        save_steps=config.get("save_steps", 100),
        load_best_model_at_end=config.get("load_best_model_at_end", True),
        metric_for_best_model=config.get("metric_for_best_model", "f1_macro"),
        greater_is_better=True,
        report_to="none",
        max_grad_norm=config.get("max_grad_norm", 2.0),
    )

    # 4. Setup W&B
    run_name = f"run_{now_str}"
    wandb_cfg = _WandbConfig()
    wandb_cfg.PROJECT_NAME = config["wb_project"]

    run_cfg = {
        **config,
        "train_start_time": now_str,
        "output_dir": out_dir,
        "training_args": training_args,
    }

    WanDBHandler.init_wandb(
        wandb_cfg, run_cfg, training_args=training_args, run_name=run_name
    )

    # 5. Trainer Setup
    callbacks = [WandbLoggingCallback(WanDBHandler)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(processor.tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Rozpoczęcie treningu
    trainer.train()

    # Zapisujemy najlepszy model do osobnego katalogu final_model
    final_model_dir = os.path.join(out_dir, "final_model")
    trainer.save_model(final_model_dir)
    processor.tokenizer.save_pretrained(final_model_dir)

    print(f"✅ Najlepszy model został zapisany w: {final_model_dir}")

    WanDBHandler.finish_wand()


if __name__ == "__main__":
    main(
        "config/training/kpwr-ner-config.json",
        "dataset/kpwr/converted/generalised/kpwr-ner-general-whole.jsonl",
    )
