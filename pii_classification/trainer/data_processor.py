import json
from typing import List, Dict, Tuple
from transformers import AutoTokenizer


class AnnoDataProcessor:
    def __init__(self, model_name: str, max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def create_label_mappings(self, data: List[Dict]) -> Tuple[Dict, Dict]:
        unique_labels = set()
        for item in data:
            unique_labels.update(item["labels"])

        # Ensure 'O' is usually the first label
        sorted_labels = sorted(list(unique_labels))
        if "O" in sorted_labels:
            sorted_labels.remove("O")
            sorted_labels = ["O"] + sorted_labels

        label2id = {label: i for i, label in enumerate(sorted_labels)}
        id2label = {i: label for label, i in label2id.items()}
        return label2id, id2label

    def tokenize_and_align_labels(self, examples: Dict, label2id: Dict) -> Dict:
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Ignore special tokens
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id.get(label[word_idx], -100))
                else:
                    # For sub-words, we usually label them as -100 or repeat the label
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def load_jsonl(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data
