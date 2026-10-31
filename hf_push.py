"""Publica o pipeline v2 de churn no Hugging Face Hub.

Uso:
    HF_TOKEN=seu_token python hf_push.py
    HF_REPO_ID=seu_usuario/neostream-churn-classifier HF_TOKEN=... python hf_push.py

Requer: huggingface_hub (instalado no .venv do workspace).
Token de escrita: https://huggingface.co/settings/tokens (tipo "Write").
"""
import os
import sys

from huggingface_hub import HfApi

REPO_ID = os.environ.get("HF_REPO_ID", "SEU_USERNAME/neostream-churn-classifier")
ARQUIVO_MODELO = "pipeline_final_churn_v2.joblib"
ARQUIVO_CARD = "model_card.md"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            "Defina HF_TOKEN (huggingface.co/settings/tokens -> Create token -> Write)."
        )
    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=ARQUIVO_MODELO,
        path_in_repo=ARQUIVO_MODELO,
        repo_id=REPO_ID,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=ARQUIVO_CARD,
        path_in_repo="README.md",  # o card vira o README do repo
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"Publicado: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
