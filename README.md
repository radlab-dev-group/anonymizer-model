# PII Classification Model

This project provides a complete pipeline for Named Entity Recognition (NER) focused on identifying Personally
Identifiable Information (PII) in Polish text. It includes utilities for data conversion, label generalization, model
training using Hugging Face Transformers, and a deployment-ready Flask API with a simple web interface.

**Repository**:
[https://github.com/radlab-dev-group/anonymizer-model](https://github.com/radlab-dev-group/anonymizer-model)

## Features

- **Data Processing CLI**: Tools to convert CONLL/IOB formats to JSONL, generalize labels, and generate distribution
  reports.
- **Training Pipeline**: A configurable trainer based on `AutoModelForTokenClassification` with Weights & Biases (W&B)
  integration.
- **Advanced Inference**: A predictor that handles sub-token merging, punctuation cleaning, and gap preservation to
  return human-readable entities.
- **REST API**: A Flask-based service to serve multiple model versions with optional dynamic quantization for faster
  inference.
- **Web Tester**: A lightweight HTML/JS interface for real-time PII detection testing.

## Installation

Ensure you have Python $\ge$ 3.9 installed.

```textmate
git clone https://github.com/radlab-dev-group/anonymizer-model.git
cd anonymizer-model
pip install .
```

## Data Preparation

The project is designed to work with datasets like `clarin-pl/kpwr-ner`.

1. **Download Dataset**: Store IOB files in `dataset/kpwr/raw/`.
2. **Convert to JSONL**:

```textmate
pii-classifier convert \
     -i dataset/kpwr/raw/kpwr-ner-n82-train-tune.iob \
     -o dataset/kpwr/converted/specific/kpwr-ner-n82-train-tune.jsonl
```

3. **Generalize Labels**:
   Map fine-grained labels to general categories using a mapping file (e.g., `config/mappings/kpwr-ner.json`).

```textmate
pii-classifier generalise \
     -i dataset/kpwr/converted/specific/kpwr-ner-n82-train-tune.jsonl \
     dataset/kpwr/converted/specific/kpwr-ner-n82-test.jsonl \
     -m config/mappings/kpwr-ner.json \
     -o dataset/kpwr/converted/generalised/kpwr-ner-general-whole.jsonl
```

4. **Generate Report**:
   Create an Excel report to analyze class distribution.

```textmate
pii-classifier report \
     -i dataset/kpwr/converted/generalised/kpwr-ner-general-whole.jsonl \
     -o dataset/kpwr/converted/generalised/kpwr-ner-general-whole-report.xlsx
```

## Training

Training is driven by a JSON configuration file located in `config/training/`.

To start training, run the training script:

```textmate
python pii_classification/trainer/train.py
```

**Key Training Features:**

- **Configurable**: Hyperparameters (learning rate, batch size, epochs) are managed via `kpwr-ner-config.json`.
- **W&B Integration**: Logs metrics and hyperparameters to Weights & Biases.
- **Automatic Export**: Saves the best model (based on `f1_macro`) into a `final_model` directory.

## Inference & API

### Running the API

The API allows you to load multiple model versions and perform predictions.

```textmate
python pii_classification/api/app.py
```

**Endpoints:**

- `GET /models`: Returns a list of available models and the default model.
- `POST /predict`: Accepts JSON with `text` and optional `model` name. Returns a list of tokens and their predicted PII
  labels.

### Web Interface

Open `pii_classification/ui/index.html` in a browser to interact with the API. The UI allows you to select a model,
input Polish text, and see highlighted PII entities.

## CLI Reference

The `pii-classifier` command provides the following sub-commands:

| Command      | Description                        | Required Arguments                                |
|:-------------|:-----------------------------------|:--------------------------------------------------|
| `convert`    | Convert CONLL/IOB file to JSONL    | `-i` (input), `-o` (output)                       |
| `generalise` | Map labels using a JSON map        | `-i` (input files), `-m` (mapping), `-o` (output) |
| `report`     | Generate Excel distribution report | `-i` (input files), `-o` (output)                 |

## Project Structure

```plain text
├── config/
│   ├── mappings/       # Label mapping JSONs
│   └── training/       # Training hyperparameter configs
├── pii_classification/
│   ├── analysis/       # Label generalization and reporting logic
│   ├── api/            # Flask API implementation
│   ├── cli/            # CLI entry point
│   ├── converters/     # Format conversion utilities
│   ├── inference/      # Model prediction and post-processing logic
│   ├── trainer/        # Training scripts and data processors
│   └── ui/             # Frontend tester (HTML/JS)
└── pyproject.toml      # Project dependencies and metadata
```
