import shutil
import torch
import argparse

from pathlib import Path
from typing import Final
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_OPSET: Final[int] = 17
DEFAULT_MAX_LENGTH: Final[int] = 128
ONNX_MODEL_FILENAME: Final[str] = "model.onnx"
NO_QUANT_DIRNAME: Final[str] = "onnx-no-quant"
QUANT_DIRNAME: Final[str] = "onnx-quant"

TOKENIZER_FILES: Final[tuple[str, ...]] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Export a local Transformers/PyTorch sequence classification model "
            "to ONNX on CPU, with optional INT8 dynamic quantization."
        )
    )

    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to a local Transformers/PyTorch model directory.",
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"ONNX opset version. Default: {DEFAULT_OPSET}.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Maximum sequence length for the sample input. Default: {DEFAULT_MAX_LENGTH}.",
    )

    parser.add_argument(
        "--dummy-text",
        type=str,
        default="To jest przykładowy tekst użyty do eksportu ONXX.",
        help="Text used as the sample input for ONNX export.",
    )

    return parser.parse_args()


def validate_model_directory(model_path: Path) -> Path:
    """Validate and return an absolute model directory path."""
    resolved_path = model_path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"The provided model path does not exist: {resolved_path}"
        )

    if not resolved_path.is_dir():
        raise NotADirectoryError(
            f"The provided model path is not a directory: {resolved_path}"
        )

    return resolved_path


def build_output_paths(model_path: Path) -> tuple[Path, Path]:
    """Build output directories for non-quantized and quantized ONNX models."""
    return model_path / NO_QUANT_DIRNAME, model_path / QUANT_DIRNAME


def create_sample_inputs(
    tokenizer: AutoTokenizer,
    dummy_text: str,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample input tensors used during ONNX export."""
    encoded_inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    input_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)

    return input_ids, attention_mask


def export_to_onnx(
    model_path: Path,
    output_dir: Path,
    opset: int,
    max_length: int,
    dummy_text: str,
) -> Path:
    """Export a Transformers/PyTorch model to ONNX."""
    device = torch.device("cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / ONNX_MODEL_FILENAME

    print(f"Loading model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    model.eval()
    model.to(device)

    print("Preparing sample input...")
    input_ids, attention_mask = create_sample_inputs(
        tokenizer=tokenizer,
        dummy_text=dummy_text,
        max_length=max_length,
        device=device,
    )

    print(f"Exporting model to ONNX: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {
                    0: "batch_size",
                    1: "sequence_length",
                },
                "attention_mask": {
                    0: "batch_size",
                    1: "sequence_length",
                },
                "logits": {
                    0: "batch_size",
                },
            },
            opset_version=opset,
            do_constant_folding=True,
        )

    print("Saving tokenizer...")
    tokenizer.save_pretrained(str(output_dir))

    print(f"Non-quantized ONNX model saved in: {output_dir}")

    return onnx_path


def quantize_onnx_model(input_onnx_path: Path, output_dir: Path) -> Path:
    """Apply dynamic INT8 quantization to an ONNX model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_onnx_path = output_dir / ONNX_MODEL_FILENAME

    print("Quantizing ONNX model to INT8...")
    print(f"Input:  {input_onnx_path}")
    print(f"Output: {output_onnx_path}")

    quantize_dynamic(
        model_input=str(input_onnx_path),
        model_output=str(output_onnx_path),
        weight_type=QuantType.QInt8,
    )

    print(f"Quantized ONNX model saved in: {output_dir}")

    return output_onnx_path


def copy_tokenizer_files(source_dir: Path, target_dir: Path) -> None:
    """Copy tokenizer files to the quantized model directory."""
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in TOKENIZER_FILES:
        source_path = source_dir / filename
        target_path = target_dir / filename

        if source_path.exists():
            shutil.copy2(source_path, target_path)


def print_header(
    model_path: Path,
    onnx_no_quant_dir: Path,
    onnx_quant_dir: Path,
) -> None:
    """Print conversion summary before execution."""
    separator = "=" * 38

    print(separator)
    print("Transformers/PyTorch -> ONNX conversion")
    print(separator)
    print(f"Source model:              {model_path}")
    print(f"Non-quantized ONNX output: {onnx_no_quant_dir}")
    print(f"Quantized ONNX output:     {onnx_quant_dir}")
    print(separator)


def print_footer(
    onnx_no_quant_dir: Path,
    onnx_quant_dir: Path,
) -> None:
    """Print final output paths."""
    separator = "=" * 38

    print(separator)
    print("Done.")
    print(f"Non-quantized ONNX: {onnx_no_quant_dir / ONNX_MODEL_FILENAME}")
    print(f"Quantized ONNX:     {onnx_quant_dir / ONNX_MODEL_FILENAME}")
    print(separator)


def main() -> None:
    """Run the ONNX export and quantization pipeline."""
    args = parse_args()

    model_path = validate_model_directory(args.model_path)
    onnx_no_quant_dir, onnx_quant_dir = build_output_paths(model_path)

    print_header(
        model_path=model_path,
        onnx_no_quant_dir=onnx_no_quant_dir,
        onnx_quant_dir=onnx_quant_dir,
    )

    onnx_path = export_to_onnx(
        model_path=model_path,
        output_dir=onnx_no_quant_dir,
        opset=args.opset,
        max_length=args.max_length,
        dummy_text=args.dummy_text,
    )

    quantize_onnx_model(
        input_onnx_path=onnx_path,
        output_dir=onnx_quant_dir,
    )

    copy_tokenizer_files(
        source_dir=onnx_no_quant_dir,
        target_dir=onnx_quant_dir,
    )

    print_footer(
        onnx_no_quant_dir=onnx_no_quant_dir,
        onnx_quant_dir=onnx_quant_dir,
    )


if __name__ == "__main__":
    main()
