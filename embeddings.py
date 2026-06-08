"""
Embedding model singleton — direct port from bare_NutriChatbot/embeddings.py.
Supports base BAAI/bge-m3 or an optional LoRA adapter if EMBEDDING_ADAPTER_PATH is set.
"""
import json
import os

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_ADAPTER_PATH = os.getenv("EMBEDDING_ADAPTER_PATH", "")

_embedding_function = None


def _load_lora_embedding(adapter_dir: str):
    """Load base model + LoRA adapter, wrap as a LangChain-compatible Embeddings object."""
    import torch
    from langchain_core.embeddings import Embeddings
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer

    config_path = os.path.join(adapter_dir, "nutribot_adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Adapter config not found at {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    base_model_name = cfg["base_model"]
    max_seq_len = cfg.get("max_seq_len", 512)
    print(f"Loading LoRA embedding: {base_model_name} + adapter from {adapter_dir}")

    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    lf_kwargs = {"local_files_only": True}

    word_embedding = Transformer(
        base_model_name, max_seq_length=max_seq_len,
        model_args=lf_kwargs, tokenizer_args=lf_kwargs,
    )
    from transformers import AutoModel
    base_hf = AutoModel.from_pretrained(
        base_model_name, trust_remote_code=True,
        dtype=torch.float32, local_files_only=True,
    )
    peft_model = PeftModel.from_pretrained(
        base_hf, adapter_dir, local_files_only=True, map_location=device
    )
    peft_model = peft_model.to(device).eval()
    word_embedding.auto_model = peft_model

    pooling_mode = cfg.get("pooling", "mean")
    pooling_kwargs = (
        {"pooling_mode_lasttoken": True}
        if pooling_mode == "lasttoken"
        else {"pooling_mode_mean_tokens": True}
    )
    pooling = Pooling(word_embedding.get_word_embedding_dimension(), **pooling_kwargs)
    st_model = SentenceTransformer(modules=[word_embedding, pooling], device=device).eval()

    class LoRAEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return st_model.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text):
            prefix = (
                "Instruct: Given a medical nutrition question, "
                "retrieve relevant passages that answer the question\nQuery: "
            )
            return st_model.encode([prefix + text], normalize_embeddings=True)[0].tolist()

    print("LoRA embedding model loaded.")
    return LoRAEmbeddings()


def get_embedding_function():
    global _embedding_function
    if _embedding_function is None:
        adapter_dir = os.path.expanduser(EMBEDDING_ADAPTER_PATH) if EMBEDDING_ADAPTER_PATH else ""
        if adapter_dir and os.path.isdir(adapter_dir):
            try:
                _embedding_function = _load_lora_embedding(adapter_dir)
                return _embedding_function
            except Exception as e:
                print(f"Warning: LoRA adapter failed ({e}); falling back to base model.")

        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("Embedding model loaded.")
    return _embedding_function
