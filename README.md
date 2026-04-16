Download dataset `clarin-pl/kpwr-ner` and store iob files to `dataset/kpwr/raw/` 

Convert files (train and test)
```shell
pii-classifier convert \
  -i dataset/kpwr/raw/kpwr-ner-n82-train-tune.iob \
  -o dataset/kpwr/converted/specific/kpwr-ner-n82-train-tune.jsonl
  
pii-classifier convert \
  -i dataset/kpwr/raw/kpwr-ner-n82-test.iob \
  -o dataset/kpwr/converted/specific/kpwr-ner-n82-test.jsonl
```

Generalise labels:
```shell
pii-classifier generalise \
  -i dataset/kpwr/converted/specific/kpwr-ner-n82-train-tune.jsonl \
  dataset/kpwr/converted/specific/kpwr-ner-n82-test.jsonl \
  -m config/mappings/kpwr-ner.json \
  -o dataset/kpwr/converted/generalised/kpwr-ner-genral-whole.jsonl
```