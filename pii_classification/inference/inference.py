# -*- coding: utf-8 -*-
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Any


class AnonPredictor:
    """
    Token‑classification predictor with optional post‑processing steps.
    """

    def __init__(self, model_path: str, use_quantized: bool = False):
        # Tokenizer & label map
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        with open(f"{model_path}/id2label.json", "r", encoding="utf-8") as f:
            self.id2label = json.load(f)

        # Load (optionally quantized) model
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.eval()
        self.model = (
            torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            if use_quantized
            else model
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def predict(
        self,
        text: str,
        clean_punct: bool = True,
        merge_entities: bool = True,
        handle_gaps: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Run inference on ``text`` and optionally apply post‑processing.

        Parameters
        ----------
        text: str
            Raw input string.
        clean_punct: bool
            If ``True`` strip leading/trailing punctuation from entity tokens.
        merge_entities: bool
            If ``True`` merge adjacent tokens that share the same label
            (respecting whitespace gaps).
        handle_gaps: bool
            If ``True`` preserve gaps (whitespace, newlines) between tokens
            when building the initial alignment.

        Returns
        -------
        List[Dict[str, str]]
            A list of ``{'word': <token>, 'label': <entity|O>}`` dictionaries.
        """
        # Tokenization & model inference
        tokens, offsets, word_ids = self._tokenize(text)

        # Align raw predictions with the original string
        aligned = self._align_predictions(
            text, tokens, offsets, word_ids, handle_gaps=handle_gaps
        )

        # Optional punctuation cleaning
        if clean_punct:
            aligned = self._clean_punctuation(aligned)

        # Optional merging of adjacent entities
        if merge_entities:
            aligned = self._merge_adjacent(aligned)

        return aligned

    # --------------------------------------------------------------------- #
    # Private helper methods
    # --------------------------------------------------------------------- #
    def _tokenize(self, text: str):
        """
        Tokenize ``text`` and return the token ids, offset mapping, and word ids.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_offsets_mapping=True,
        )
        offsets_tensor = inputs.pop("offset_mapping")
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Convert tensors to plain Python lists for easier handling
        tokens = predictions[0].tolist()
        offsets = offsets_tensor[0].tolist()
        word_ids = inputs.word_ids(batch_index=0)

        return tokens, offsets, word_ids

    def _align_predictions(
        self,
        text: str,
        token_ids: List[int],
        offsets: List[List[int]],
        word_ids: List[Any],
        handle_gaps: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Build a list of ``{'word': ..., 'label': ...}`` entries that respects
        the original spacing, leading whitespace, and sub‑token merging.
        """
        results = []
        last_end_char = 0

        for i, (start, end) in enumerate(offsets):
            word_id = word_ids[i]

            # Skip special tokens (CLS, SEP, padding)
            if word_id is None:
                continue

            # ---- Gap handling -------------------------------------------------
            if handle_gaps and start > last_end_char:
                gap_text = text[last_end_char:start]
                results.append({"word": gap_text, "label": "O"})

            # Normalize label (strip B-/I- prefixes)
            raw_label = self.id2label.get(str(token_ids[i]), "O")
            label = raw_label.replace("B-", "").replace("I-", "")

            word_text = text[start:end]

            # ---- Leading whitespace for RoBERTa / XLM‑R -----------------------
            if word_id != (word_ids[i - 1] if i > 0 else None):
                stripped_word = word_text.lstrip()
                whitespace = word_text[: len(word_text) - len(stripped_word)]
                if whitespace:
                    results.append({"word": whitespace, "label": "O"})
                    word_text = stripped_word

            # ---- Sub‑token merging --------------------------------------------
            if word_id == (word_ids[i - 1] if i > 0 else None):
                # Extend previous token
                results[-1]["word"] += word_text
            else:
                results.append({"word": word_text, "label": label})

            last_end_char = end

        # Trailing whitespace (if any)
        if last_end_char < len(text):
            results.append({"word": text[last_end_char:], "label": "O"})

        return results

    @staticmethod
    def _clean_punctuation(token_seq: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Strip leading / trailing punctuation from non‑``O`` tokens.
        Punctuation is emitted as separate ``O`` tokens so that it
        acts as a barrier for later merging.
        """
        cleaned = []
        punctuation = ".,!?;:)]}"

        for item in token_seq:
            if item["label"] != "O" and item["word"]:
                word = item["word"]
                label = item["label"]
                leading, trailing = "", ""

                # Extract leading punctuation
                while word and word[0] in punctuation:
                    leading += word[0]
                    word = word[1:]

                # Extract trailing punctuation
                while word and word[-1] in punctuation:
                    trailing = word[-1] + trailing
                    word = word[:-1]

                if leading:
                    cleaned.append({"word": leading, "label": "O"})
                if word:
                    cleaned.append({"word": word, "label": label})
                if trailing:
                    cleaned.append({"word": trailing, "label": "O"})
            else:
                cleaned.append(item)

        return cleaned

    @staticmethod
    def _merge_adjacent(token_seq: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Merge consecutive tokens that share the same non‑``O`` label.
        Whitespace‑only ``O`` tokens are merged **only** when they sit
        between two tokens of the same label.
        """
        if not token_seq:
            return token_seq

        merged = []
        i = 0
        while i < len(token_seq):
            cur = token_seq[i]

            # Initialise merged list
            if not merged:
                merged.append(cur)
                i += 1
                continue

            prev = merged[-1]

            # Case A: Same non‑O label → concatenate
            if cur["label"] == prev["label"] and cur["label"] != "O":
                prev["word"] += cur["word"]
                i += 1
                continue

            # Case B: Current token is pure whitespace O and next token matches prev label
            if (
                cur["label"] == "O"
                and cur["word"].strip() == ""
                and i + 1 < len(token_seq)
                and token_seq[i + 1]["label"] == prev["label"]
                and prev["label"] != "O"
            ):
                # Merge whitespace + next token into previous entity
                prev["word"] += cur["word"] + token_seq[i + 1]["word"]
                i += 2  # skip current + next
                continue

            # Otherwise just append
            merged.append(cur)
            i += 1

        return merged
