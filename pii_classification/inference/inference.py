import json
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification


class AnonPredictor:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        with open(f"{model_path}/id2label.json", "r") as f:
            self.id2label = json.load(f)

        self.model.eval()

    def predict(self, text: str):
        # Tokenize input with offset_mapping to preserve original formatting
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
        )

        # Extract offset_mapping so it's not passed to the model
        offsets_tensor = inputs.pop("offset_mapping")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        # ==========================================================================
        # Decode labels and align with words
        predicted_ids = predictions[0].tolist()
        offsets = offsets_tensor[0].tolist()
        word_ids = inputs.word_ids(batch_index=0)

        results = []
        last_end_char = 0

        for i, (start, end) in enumerate(offsets):
            word_id = word_ids[i]

            # Skip special tokens (like [CLS], [SEP])
            if word_id is None:
                continue

            # 1. Handle gaps (whitespace, newlines) between tokens
            if start > last_end_char:
                gap_text = text[last_end_char:start]
                results.append({"word": gap_text, "label": "O"})

            # Normalize label: remove B- and I- prefixes for easier merging and display
            raw_label = self.id2label.get(str(predicted_ids[i]), "O")
            label = raw_label.replace("B-", "").replace("I-", "")

            word_text = text[start:end]

            # 2. Handle RoBERTa/XLM-R leading whitespace
            # If this is the start of a new word,
            # check if the token starts with whitespace
            if word_id != (word_ids[i - 1] if i > 0 else None):
                stripped_word = word_text.lstrip()
                whitespace = word_text[: len(word_text) - len(stripped_word)]
                if whitespace:
                    # The leading space should be 'O', not part of the entity
                    results.append({"word": whitespace, "label": "O"})
                    word_text = stripped_word

            # 3. Handle sub-tokens: merge them into one word
            if word_id == (word_ids[i - 1] if i > 0 else None):
                # Append to the last added word (which might be a sub-token
                # or the stripped word) We find the last element
                # that corresponds to the current word_id
                results[-1]["word"] += word_text
            else:
                results.append({"word": word_text, "label": label})

            last_end_char = end

        # Handle any remaining trailing whitespace
        if last_end_char < len(text):
            results.append({"word": text[last_end_char:], "label": "O"})

        # ==========================================================================
        # Merge adjacent identical labels
        if not results:
            return results

        merged = []
        i = 0
        while i < len(results):
            curr = results[i]
            if not merged:
                merged.append(curr)
                i += 1
                continue

            prev = merged[-1]

            # Case 1: Same label (not 'O') -> Merge immediately
            if curr["label"] == prev["label"] and curr["label"] != "O":
                prev["word"] += curr["word"]
                i += 1
            # Case 2: Current is 'O' (whitespace/punctuation)
            # and the NEXT is the same label as prev
            elif (
                i + 1 < len(results)
                and curr["label"] == "O"
                and curr["word"].strip() == ""
                and results[i + 1]["label"] == prev["label"]
                and prev["label"] != "O"
            ):

                # Absorb the gap and the next token into the previous entity
                prev["word"] += curr["word"] + results[i + 1]["word"]

                # Skip current 'O' and the next merged token
                i += 2
            else:
                merged.append(curr)
                i += 1
        # ==========================================================================
        # Clean up trailing punctuation from entities
        final_results = []
        # Define punctuation that should not be part of an entity if it's at the end
        trailing_punct = ".,!?;:)]}"

        for item in merged:
            if item["label"] != "O" and item["word"]:
                word = item["word"]
                punct_part = ""

                # Extract all trailing punctuation characters
                while word and word[-1] in trailing_punct:
                    punct_part = word[-1] + punct_part
                    word = word[:-1]

                if punct_part:
                    # Add the cleaned word as the entity
                    if word:
                        final_results.append({"word": word, "label": item["label"]})
                    # Add the punctuation as 'O'
                    final_results.append({"word": punct_part, "label": "O"})
                else:
                    final_results.append(item)
            else:
                final_results.append(item)

        return final_results
