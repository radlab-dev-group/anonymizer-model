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
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

from data_processor import NERDataProcessor
from rdl_ml_utils.handlers.wandb_handler import WanDBHandler


class _WandbConfig:
    PROJECT_NAME: str = os.getenv("WANDB_PROJECT", "kpwr-ner")
    PROJECT_TAGS: list = (
        os.getenv("WANDB_TAGS", "").split(",") if os.getenv("WANDB_TAGS") else []
    )
    PREFIX_RUN: str = os.getenv("WANDB_PREFIX", "run_")
    BASE_RUN_NAME: str = os.getenv("WANDB_BASE", "ner-model")


class WandbLoggingCallback(TrainerCallback):
    """Callback przesyłający logi z Trainer-a do W&B za pomocą WanDBHandler."""

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

    report = classification_report(true_labels, true_predictions, output_dict=True)

    # Logujemy zarówno średnie Macro, jak i Weighted, aby mieć pełny obraz precyzji i recallu
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

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(config["output_dir"], now_str)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Data Processing (potrzebne do ustalenia num_labels przed TrainingArguments)
    processor = NERDataProcessor(config["model_name"], config["max_length"])
    raw_data = processor.load_jsonl(data_path)
    label2id, id2label = processor.create_label_mappings(raw_data)

    # Save mappings
    with open(os.path.join(out_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)
    with open(os.path.join(out_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f)

    # Prepare Dataset
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

    # Global map for metrics
    global label_map
    label_map = id2label

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # Wyłączamy domyślny raport, bo używamy WanDBHandler
        logging_strategy="step",
        logging_steps=10,
    )

    # 4. Setup W&B via WanDBHandler
    run_name = f"run_{now_str}"
    wandb_cfg = _WandbConfig()
    # Dodajemy training_args do konfiguracji run_cfg, aby były widoczne w W&B
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
    # Przekazujemy klasę WanDBHandler jako handler do callbacka
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

    trainer.train()

    # Save final model
    final_model_dir = os.path.join(out_dir, "final_model")
    trainer.save_model(final_model_dir)
    processor.tokenizer.save_pretrained(final_model_dir)

    # Zamykamy sesję WandB
    WanDBHandler.finish_wand()


if __name__ == "__main__":
    main(
        "config/training/kpwr-ner-config.json",
        "dataset/kpwr/converted/generalised/kpwr-ner-general-whole.jsonl",
    )
