1. Download datasets:
 a. clarin-pl/kpwr-ner
2. Convert datasets to common jsonl format
  a. kpwr -> using conll2jsonl.py
3. Genralisation: map the dataset with specific mappings file
  a. kpwr -> using jsonl_generalise_labels.json and `mappings/kpwr-ner.json`