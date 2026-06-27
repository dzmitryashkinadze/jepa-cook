import torch
import torch.nn as nn
from transformers import AutoTokenizer


def tokenize_batch(texts: list[str], tokenizer, max_len: int = 128) -> torch.Tensor:
    """Tokenizes a flat list of strings into padded index sequence tensors."""
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return encoded["input_ids"]


def tokenize_nested(groups: list[list[str]], tokenizer, max_len: int = 128) -> torch.Tensor:
    """Tokenizes a nested array of items into [batch, num_elements, max_len] dimensions."""
    # Since we are executing 1 batch instance inference, batch_size = 1
    num_elements = len(groups[0])

    flat_texts = [item for group in groups for item in group]
    flat_tokens = tokenize_batch(flat_texts, tokenizer, max_len)

    return flat_tokens.view(1, num_elements, max_len)


def main():
    # 1. Target configurations and Hugging Face paths
    repo_id = "DzmitryAshkinadze/jepa-cook"
    tokenizer_name = "sentence-transformers/all-MiniLM-L6-v2"
    max_len = 128

    # Input mock variables passed directly from your example evaluation run
    ingredients_input = ["cream", "egg yolks", "sugar"]
    actions_input = ["whisk over water bath until thickened and chill"]
    targets_input = ["ice cream", "custard", "cream", "scrambled eggs"]

    print(f"Fetching and caching model files from Hub ID: {repo_id}...")

    from huggingface_hub import hf_hub_download  # deptry: ignore

    from jepa_cook.src.config import RecipeJEPAConfig  # deptry: ignore
    from jepa_cook.src.models import RecipeJEPA  # deptry: ignore

    # 1. Download configuration and construct the architecture framework
    config = RecipeJEPAConfig.from_pretrained(repo_id)
    model = RecipeJEPA(config)

    # 2. Fetch only the pure binary weights file path from your HF cache local path
    weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")

    # 3. Standard PyTorch state dictionary injection (completely bypasses buggy HF hooks)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print("Model weights successfully initialized via standard state dictionary mapping!")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.eval()

    # 3. Process structural strings into tensors compatible with JEPA layers
    # Enclosing inputs to match training structural expected shapes [Batch, Count, Length]
    x_tokens = tokenize_nested([ingredients_input], tokenizer, max_len=max_len)
    a_tokens = tokenize_nested([actions_input], tokenizer, max_len=max_len)

    # 4. Process targets and extract their representation embeddings
    y_tokens = tokenize_batch(targets_input, tokenizer, max_len=max_len)

    print("\nRunning predictive architecture execution passes...")
    with torch.no_grad():
        # Get forward predictor projections
        pred_embed, _, _, _ = model(x_tokens, a_tokens)  # Outputs shape [1, latent_dim]

        # Extract target vectors using target encoder lookup
        target_embeds = model.encode_target(y_tokens)  # Outputs shape [4, latent_dim]

    # 5. Apply L2 Unit-Normalization evaluation metrics over vectors
    # Normalizing along vector dimensions
    pred_embed_norm = nn.functional.normalize(pred_embed, p=2, dim=-1)
    target_embeds_norm = nn.functional.normalize(target_embeds, p=2, dim=-1)

    print("\n" + "=" * 60)
    print(" EVALUATION WITH L2 UNIT-NORMALIZATION (REMOTE HUB INF)")
    print("=" * 60)

    # Compute MSE distance metrics between predicted vectors and structural candidates
    for i, target_name in enumerate(targets_input):
        single_target_norm = target_embeds_norm[i : i + 1]

        # Calculate standard Mean Squared Error over normalized coordinates
        mse_loss = nn.functional.mse_loss(pred_embed_norm, single_target_norm)

        print(f" {target_name:<30} | Normalized MSE: {mse_loss.item():.6f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
