"""Load local or Hugging Face models for attention-based self-coding."""

from pathlib import Path


def load_model(
    model_path: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
    device: str = "auto",
    dtype: str = "auto",
    revision: str = "main",
    trust_remote_code: bool = False,
    local_files_only: bool = False,
):
    # Lazy imports keep prompt preparation and cache export usable without torch.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    shared = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path or model_path), **shared
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_key = (
        "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto" if device == "auto" else {"": device},
        attn_implementation="eager",
        **{dtype_key: "auto" if dtype == "auto" else getattr(torch, dtype)},
        **shared,
    ).eval()
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
