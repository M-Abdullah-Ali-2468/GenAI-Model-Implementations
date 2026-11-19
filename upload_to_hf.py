import argparse
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    ckpt_dir = ROOT / "checkpoints"
    assert ckpt_dir.exists(), "checkpoints/ folder not found"

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(folder_path=str(ckpt_dir), repo_id=args.repo_id, repo_type="model")
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
