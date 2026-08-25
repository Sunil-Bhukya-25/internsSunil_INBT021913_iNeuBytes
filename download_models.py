from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "Sunil-25-10/personalized-chatbot-models"
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PERSONALIZED CHATBOT — MODEL DOWNLOAD")
print("=" * 70)
print("Repository:", REPO_ID)
print("Target:", MODEL_DIR.resolve())
print()

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=str(MODEL_DIR),
)

required = [
    "tfidf_vectorizer.pkl",
    "tfidf_train_matrix.npz",
    "semantic_train_embeddings_fixed.npy",
    "semantic_train_indices_fixed.npy",
    "semantic_train_responses_fixed.json",
    "semantic_embedding_config_fixed.json",
]

missing = [name for name in required if not (MODEL_DIR / name).exists()]

if missing:
    print("ERROR: Missing required model files:")
    for name in missing:
        print(" -", name)
    raise SystemExit(1)

print()
print("SUCCESS — all required retrieval artifacts are present.")
for name in required:
    print("OK:", name)
print("=" * 70)
