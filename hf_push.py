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

# O namespace (username) é descoberto automaticamente pelo token via whoami();
# HF_REPO_ID permite customizar o nome do repo (ex.: "meu_usuario/outro-nome").
ARQUIVO_MODELO = "pipeline_final_churn_v2.joblib"
ARQUIVO_CARD = "model_card.md"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            "Defina HF_TOKEN (huggingface.co/settings/tokens -> Create token -> Write)."
        )
    api = HfApi(token=token)

    # o username vem do próprio token; HF_REPO_ID só para customizar o nome
    usuario = api.whoami()["name"]
    repo_id = os.environ.get("HF_REPO_ID", f"{usuario}/neostream-churn-classifier")
    if "SEU_USERNAME" in repo_id:
        sys.exit("Defina HF_REPO_ID com seu username real (ex.: renefranciscofs/...).")

    print(f"Publicando em {repo_id} (usuário: {usuario}) ...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=ARQUIVO_MODELO,
        path_in_repo=ARQUIVO_MODELO,
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=ARQUIVO_CARD,
        path_in_repo="README.md",  # o card vira o README do repo
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Publicado: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
