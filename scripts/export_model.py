#!/usr/bin/env python3
"""
Export sentence-transformers/all-MiniLM-L6-v2 to ONNX format.

Requires: pip install optimum[onnxruntime] transformers sentence-transformers

Outputs:
  models/model.onnx  - ONNX model (transformer body, last_hidden_state output)
  models/vocab.txt   - WordPiece vocabulary for tokenizer
"""
import os
import json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    print(f"Loading tokenizer from {model_id}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    vocab_path = os.path.join(OUTPUT_DIR, "vocab.txt")
    print(f"Saving vocab ({len(tokenizer.vocab)} tokens) to {vocab_path}")
    tokenizer.save_vocabulary(OUTPUT_DIR)

    print(f"Exporting {model_id} to ONNX...")
    from transformers import AutoModel
    import torch

    model = AutoModel.from_pretrained(model_id)

    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 16), dtype=torch.int64)
    dummy_attention_mask = torch.ones(1, 16, dtype=torch.int64)
    dummy_token_type_ids = torch.zeros(1, 16, dtype=torch.int64)

    model_path = os.path.join(OUTPUT_DIR, "model.onnx")
    print(f"Exporting to {model_path}...")

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
        model_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    print(f"Done! Files in {OUTPUT_DIR}:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f} ({size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
