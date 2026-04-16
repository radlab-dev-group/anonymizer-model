import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification


class AnonPredictor:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        with open(f"{model_path}/id2label.json", "r") as f:
            self.id2label = json.load(f)

        self.model.eval()

    def predict(self, text: str):
        # Tokenize input
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Decode labels and align with words
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_ids = predictions[0].tolist()

        results = []
        # Use word_ids to group sub-tokens back into words
        word_ids = inputs.word_ids(batch_index=0)

        current_word = None
        current_label = None

        for token, pred_id, word_id in zip(tokens, predicted_ids, word_ids):
            if word_id is None:
                continue

            label = self.id2label.get(str(pred_id), "O")

            if word_id != current_word:
                if current_word is not None:
                    results.append(
                        {"word": current_word_text, "label": current_label}
                    )
                current_word = word_id
                current_word_text = token.replace("##", "")
                current_label = label
            else:
                current_word_text += token.replace("##", "")
                # We keep the label of the first sub-token (B- or I- logic)

        # Append the last word
        if current_word is not None:
            results.append({"word": current_word_text, "label": current_label})

        return results
