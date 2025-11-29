"""Download the chatglm3-6b GGUF model for local inference.

This script fetches `chatglm3-6b.Q4_K_M.gguf` from the mradermacher/chatglm3-6b-GGUF
repository on Hugging Face and writes it to a local directory for use with
`llama.cpp` or the Python `llama_cpp` bindings.
"""

import argparse
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "mradermacher/chatglm3-6b-GGUF"
DEFAULT_FILENAME = "chatglm3-6b.Q4_K_M.gguf"
DEFAULT_OUTPUT_DIR = "models"


def download_model(repo_id: str, filename: str, output_dir: str, token: Optional[str], force: bool) -> str:
    """Download the GGUF file and return the local path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        force_download=force,
    )
    return str(local_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download chatglm3 GGUF model for llama.cpp usage.")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face repository ID that hosts the GGUF file.",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help="Name of the GGUF file to download from the repo.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to place the downloaded model.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token for gated or throttled downloads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the file already exists locally.",
    )

    args = parser.parse_args()

    local_path = download_model(
        repo_id=args.repo_id,
        filename=args.filename,
        output_dir=args.output_dir,
        token=args.token,
        force=args.force,
    )
    print(f"Model downloaded to: {local_path}")


if __name__ == "__main__":
    main()
