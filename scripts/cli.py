import argparse
import ast
import copy
import json
import math
import os

import polars as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

# =====================================================================
# 1. DATASETS & COLLATOR
# =====================================================================


class PreTokenizedActionDataset(Dataset):
    def __init__(self, dataset_path: str):
        # Read the structural parquet file natively
        self.df = pl.read_parquet(dataset_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.row(idx, named=True)
        # Convert lists of lists / lists directly to tensors
        x = [torch.tensor(item, dtype=torch.long) for item in row["x_tokens"]]
        a = [torch.tensor(item, dtype=torch.long) for item in row["a_tokens"]]
        y = torch.tensor(row["y_tokens"], dtype=torch.long)
        return x, a, y


def pad_nested_sequences(batch_lists, max_len=128):
    """Pads a nested batch of varied length structural elements into a 3D Tensor.

    Returns: Shape [batch_size, max_elements_in_batch, max_len]
    """
    batch_size = len(batch_lists)
    max_elements = max(len(row) for row in batch_lists)
    max_elements = max(1, max_elements)  # Safeguard empty rows

    padded_tensor = torch.zeros(batch_size, max_elements, max_len, dtype=torch.long)

    for i, row in enumerate(batch_lists):
        for j, element in enumerate(row):
            length = min(len(element), max_len)
            if length > 0:
                padded_tensor[i, j, :length] = element[:length]

    return padded_tensor


def jepa_collate_fn(batch):
    xs, as_, ys = zip(*batch)

    # Pad 3D Input structural layers
    x_tensor = pad_nested_sequences(xs, max_len=128)
    a_tensor = pad_nested_sequences(as_, max_len=128)

    # Pad 2D Target flat layer
    y_tensor = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    if y_tensor.size(1) < 128:
        padding = torch.zeros(y_tensor.size(0), 128 - y_tensor.size(1), dtype=torch.long)
        y_tensor = torch.cat([y_tensor, padding], dim=1)
    else:
        y_tensor = y_tensor[:, :128]

    return x_tensor, a_tensor, y_tensor


# =====================================================================
# 2. MODULAR ENCODING ARCHITECTURES
# =====================================================================


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)


class StructuralGroupEncoder(nn.Module):
    """Encodes a nested 3D tensor of multiple item components into a singular joint latent vector."""

    def __init__(self, embedding_layer: Embedding, embed_dim: int, latent_dim: int):
        super().__init__()
        self.embedding_layer = embedding_layer

        self.element_compressor = nn.Sequential(nn.Linear(embed_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())

        self.joint_projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens shape: [batch_size, num_elements, max_len]
        batch_size, num_elements, max_len = tokens.shape

        # 1. Flatten for uniform embedding step evaluation
        flat_tokens = tokens.view(-1, max_len)  # [B * N, L]
        mask = (flat_tokens != 0).float()
        mask_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        embeddings = self.embedding_layer(flat_tokens)  # [B * N, L, D]
        masked_embeddings = embeddings * mask.unsqueeze(-1)

        # Mean pool individual sub-elements text chunks
        pooled_elements = masked_embeddings.sum(dim=1) / mask_counts  # [B * N, D]
        element_latents = self.element_compressor(pooled_elements)  # [B * N, H]

        # 2. Re-assemble to row segments structures
        group_latents = element_latents.view(batch_size, num_elements, -1)  # [B, N, H]

        # Pool all combined elements across rows safely without dimension pollution
        group_mask = (tokens.sum(dim=-1) != 0).float()  # [B, N]
        group_counts = group_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]

        masked_group = group_latents * group_mask.unsqueeze(-1)  # [B, N, H]
        combined_latent = masked_group.sum(dim=1) / group_counts  # [B, H] / [B, 1] -> [B, H]

        return self.joint_projection(combined_latent)


class ActionSequenceEncoder(nn.Module):
    """Generates structural sequence memories from multi-step action groups for predictor decoding."""

    def __init__(self, embedding_layer: Embedding, embed_dim: int, latent_dim: int):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.compressor = nn.Sequential(nn.Linear(embed_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())

    def forward(self, a_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # a_tokens shape: [batch_size, num_actions, max_len]
        batch_size, num_actions, max_len = a_tokens.shape

        flat_actions = a_tokens.view(-1, max_len)
        mask = (flat_actions != 0).float()
        mask_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        embeddings = self.embedding_layer(flat_actions)
        masked_embeddings = embeddings * mask.unsqueeze(-1)

        pooled_actions = masked_embeddings.sum(dim=1) / mask_counts
        u_seq = self.compressor(pooled_actions).view(batch_size, num_actions, -1)  # [B, N, H]

        # Action sequence mask profile evaluation
        a_mask = (a_tokens.sum(dim=-1) != 0).float()  # [B, N]
        return u_seq, a_mask


class TitleTargetEncoder(nn.Module):
    """Processes flat 2D title targets directly."""

    def __init__(self, embedding_layer: Embedding, embed_dim: int, latent_dim: int):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens shape: [batch_size, max_len]
        mask = (tokens != 0).float()
        mask_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        embeddings = self.embedding_layer(tokens)
        masked_embeddings = embeddings * mask.unsqueeze(-1)
        pooled = masked_embeddings.sum(dim=1) / mask_counts

        return self.encoder(pooled)


class Predictor(nn.Module):
    def __init__(self, latent_dim: int = 256, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation=nn.functional.silu,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, z_t: torch.Tensor, u_seq: torch.Tensor, a_mask: torch.Tensor | None = None) -> torch.Tensor:
        tgt = z_t.unsqueeze(1)
        # Invert mask: True for elements that should be IGNORED by attention
        mem_key_padding_mask = (a_mask == 0) if a_mask is not None else None
        out = self.transformer_decoder(tgt=tgt, memory=u_seq, memory_key_padding_mask=mem_key_padding_mask)
        return out.squeeze(1)


class RecipeJEPA(nn.Module):
    def __init__(self, vocab_size: int = 30522, embed_dim: int = 384, latent_dim: int = 256):
        super().__init__()

        self.embedding = Embedding(vocab_size, embed_dim)

        # Multi-Item Encoders
        self.ingredient_encoder = StructuralGroupEncoder(self.embedding, embed_dim, latent_dim)
        self.action_encoder = ActionSequenceEncoder(self.embedding, embed_dim, latent_dim)

        # Target Title Pipeline (EMA target framework)
        self.target_embedding = copy.deepcopy(self.embedding)
        self.target_encoder = TitleTargetEncoder(self.target_embedding, embed_dim, latent_dim)

        for param in self.target_embedding.parameters():
            param.requires_grad = False
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = Predictor(latent_dim)
        self.delta_norm = nn.LayerNorm(latent_dim)
        self.action_gate = nn.Parameter(torch.tensor([0.1]))
        self.prediction_norm = nn.LayerNorm(latent_dim)

    def encode_ingredients(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.ingredient_encoder(tokens)

    def encode_target(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.target_encoder(tokens)

    def forward(self, x_tokens: torch.Tensor, a_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_t = self.encode_ingredients(x_tokens)
        u_seq, a_mask = self.action_encoder(a_tokens)

        latent_delta = self.predictor(z_t, u_seq, a_mask)
        normalized_delta = self.delta_norm(latent_delta)

        z_next_pred = self.prediction_norm(z_t + self.action_gate * normalized_delta)
        return z_next_pred, z_t

    @torch.no_grad()
    def update_target_ema(self, momentum: float = 0.99):
        for target_param, ingredient_param in zip(
            self.target_encoder.parameters(), self.ingredient_encoder.parameters()
        ):
            # Use skip matching logic on shape changes safely if internal weights differ
            if target_param.shape == ingredient_param.shape:
                target_param.data.mul_(momentum).add_(ingredient_param.data, alpha=1.0 - momentum)

        for target_embedding_param, embedding_param in zip(
            self.target_embedding.parameters(), self.embedding.parameters()
        ):
            target_embedding_param.data.mul_(momentum).add_(embedding_param.data, alpha=1.0 - momentum)


# =====================================================================
# 3. LOSS FUNCTION (VICReg)
# =====================================================================


def vicreg_loss(
    z_pred,
    z_true,
    z_t,
    sim_coeff=25.0,
    var_coeff=25.0,
    cov_coeff=5.0,
    repel_coeff=15.0,
    threshold=0.25,
    gamma=1.0,
    eps=1e-4,
):
    batch_size, num_features = z_pred.shape
    sim_loss = nn.functional.mse_loss(z_pred, z_true)

    shortcut_distance = torch.mean((z_pred - z_t) ** 2, dim=-1)
    repel_loss = torch.mean(nn.functional.relu(shortcut_distance - threshold))

    std_a = torch.sqrt(z_pred.var(dim=0) + eps)
    std_b = torch.sqrt(z_true.var(dim=0) + eps)
    var_loss_a = torch.mean(nn.functional.relu(gamma - std_a))
    var_loss_b = torch.mean(nn.functional.relu(gamma - std_b))
    std_loss = var_loss_a + var_loss_b

    z_pred_zero_mean = z_pred - z_pred.mean(dim=0)
    z_true_zero_mean = z_true - z_true.mean(dim=0)
    cov_a = (z_pred_zero_mean.T @ z_pred_zero_mean) / (batch_size - 1)
    cov_b = (z_true_zero_mean.T @ z_true_zero_mean) / (batch_size - 1)
    cov_loss_a = cov_a.pow(2).sum() - cov_a.diagonal().pow(2).sum()
    cov_loss_b = cov_b.pow(2).sum() - cov_b.diagonal().pow(2).sum()
    cov_loss = (cov_loss_a + cov_loss_b) / num_features

    total_loss = (sim_coeff * sim_loss) + (var_coeff * std_loss) + (cov_coeff * cov_loss) + (repel_coeff * repel_loss)
    return total_loss, sim_loss, std_loss, cov_loss, repel_loss


# =====================================================================
# 4. TRAINER MODULE
# =====================================================================


class JEPATrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        output_dir,
        log_dir,
        patience=5,
        min_delta=0.001,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_sim = float("inf")
        self.patience_counter = 0

    def train(self, epochs: int):
        os.makedirs(self.output_dir, exist_ok=True)
        for epoch in range(epochs):
            self.model.train()
            train_loss_monitor = [0.0, 0.0, 0.0, 0.0]

            for batch_idx, (x_tokens, a_tokens, y_tokens) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x_tokens, a_tokens, y_tokens = (
                    x_tokens.to(self.device),
                    a_tokens.to(self.device),
                    y_tokens.to(self.device),
                )

                pred_embed, z_t = self.model(x_tokens, a_tokens)
                with torch.no_grad():
                    true_embed = self.model.encode_target(y_tokens).detach()

                total_loss, sim_loss, std_loss, cov_loss, repel_loss = vicreg_loss(pred_embed, true_embed, z_t)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                self.model.update_target_ema(momentum=0.99)

                sim_loss, std_loss, cov_loss, repel_loss = (
                    sim_loss.item(),
                    std_loss.item(),
                    cov_loss.item(),
                    repel_loss.item(),
                )

                train_loss_monitor[0] += sim_loss
                train_loss_monitor[1] += std_loss
                train_loss_monitor[2] += cov_loss
                train_loss_monitor[3] += repel_loss

                self.global_step += 1

                if batch_idx % 100 == 0:
                    print(
                        f"Batch: {batch_idx}/{len(self.train_loader)} |",
                        f"SIM: {sim_loss:.4f} |",
                        f"STD: {std_loss:.4f} |",
                        f"COV: {cov_loss:.4f} |",
                        f"REPEL: {repel_loss:.4f}",
                    )

            epoch_train_sim = train_loss_monitor[0] / len(self.train_loader)
            epoch_train_std = train_loss_monitor[1] / len(self.train_loader)
            epoch_train_cov = train_loss_monitor[2] / len(self.train_loader)
            epoch_train_rep = train_loss_monitor[3] / len(self.train_loader)
            epoch_val_sim, epoch_val_std, epoch_val_cov, epoch_val_rep = self.validate()

            print(
                f"Epoch {epoch + 1:02d} |",
                f"Train SIM: {epoch_train_sim:.4f} |",
                f"STD: {epoch_train_std:.4f} |",
                f"COV: {epoch_train_cov:.4f} |",
                f"REP: {epoch_train_rep:.4f}",
            )
            print(
                f"Epoch {epoch + 1:02d} |",
                f"Val SIM: {epoch_val_sim:.4f} |",
                f"STD: {epoch_val_std:.4f} |",
                f"COV: {epoch_val_cov:.4f} |",
                f"REP: {epoch_val_rep:.4f}",
            )

            if epoch_val_sim < (self.best_val_sim - self.min_delta):
                self.best_val_sim = epoch_val_sim
                self.patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "recipe_jepa_model_best.pt"))
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print("[🛑] Early stopping triggered.")
                break
        self.writer.close()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss_monitor = [0.0, 0.0, 0.0, 0.0]
        for x_tokens, a_tokens, y_tokens in self.val_loader:
            x_tokens, a_tokens, y_tokens = x_tokens.to(self.device), a_tokens.to(self.device), y_tokens.to(self.device)
            pred_embed, z_t = self.model(x_tokens, a_tokens)
            true_embed = self.model.encode_target(y_tokens).detach()
            _, sim_loss, std_loss, cov_loss, rep_loss = vicreg_loss(pred_embed, true_embed, z_t)
            val_loss_monitor[0] += sim_loss.item()
            val_loss_monitor[1] += std_loss.item()
            val_loss_monitor[2] += cov_loss.item()
            val_loss_monitor[3] += rep_loss.item()
        return [v / len(self.val_loader) for v in val_loss_monitor]


# =====================================================================
# 5. SUBCOMMAND INTERFACES
# =====================================================================


def handle_train(args, device):
    train_dataset = PreTokenizedActionDataset(args.train_dataset)
    val_dataset = PreTokenizedActionDataset(args.val_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=jepa_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=jepa_collate_fn
    )

    model = RecipeJEPA(vocab_size=30522, embed_dim=384, latent_dim=256).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    num_warmup_steps = 3 * len(train_loader)
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    trainer = JEPATrainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        args.output_dir,
        args.log_dir,
        args.patience,
        args.min_delta,
    )
    trainer.train(epochs=args.epochs)


def handle_inference(args, device):
    try:
        targets_list = ast.literal_eval(args.targets)
    except Exception:
        print("[!] Format error parsing target list strings.")
        return

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = RecipeJEPA(vocab_size=30522, embed_dim=384, latent_dim=256)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint if "state_dict" not in checkpoint else checkpoint["state_dict"])
    model.to(device).eval()

    with torch.no_grad():
        # Clean list string extraction parsing logic to match 3D format layout
        def tokenize_to_3d_input(text_input):
            try:
                lst = json.loads(text_input) if "[" in text_input else [text_input]
            except Exception:
                lst = [text_input]
            tokens_list = [torch.tensor(tokenizer(item, add_special_tokens=False)["input_ids"]) for item in lst]
            return pad_nested_sequences([tokens_list], max_len=128).to(device)

        x_tokens = tokenize_to_3d_input(args.ingredients)
        a_tokens = tokenize_to_3d_input(args.action)

        pred_z_next, _ = model(x_tokens, a_tokens)

        # Tokenize targets array using standard flat 2D strategy
        y_enc = tokenizer(targets_list, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        y_tokens_batch = y_enc["input_ids"].to(device)

        true_z_next_batch = model.encode_target(y_tokens_batch)

        rankings = []
        for idx, target_str in enumerate(targets_list):
            true_z_next = true_z_next_batch[idx].unsqueeze(0)
            distance = nn.functional.mse_loss(pred_z_next, true_z_next).item()
            rankings.append((target_str, distance))

        rankings.sort(key=lambda x: x[1])

    print("\n" + "=" * 60)
    print(f" INITIAL STATE (s_t): {args.ingredients}")
    print(f" ACTION (a_t):        {args.action}")
    print("=" * 60)
    for target, score in rankings:
        print(f"{target:<30} | MSE: {score:.6f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Recipe JEPA Unified CLI Workflow")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Workflow Mode")

    train_parser = subparsers.add_parser("train", help="Run model training loop")
    train_parser.add_argument("--train_dataset", type=str, default="data/recipe_train.parquet")
    train_parser.add_argument("--val_dataset", type=str, default="data/recipe_val.parquet")
    train_parser.add_argument("--output_dir", type=str, default="checkpoints")
    train_parser.add_argument("--log_dir", type=str, default="runs/recipe_jepa_experiment")
    train_parser.add_argument("--batch_size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--patience", type=int, default=3)
    train_parser.add_argument("--min_delta", type=float, default=1e-4)

    infer_parser = subparsers.add_parser("inference", help="Run prediction inference")
    infer_parser.add_argument("--checkpoint", type=str, required=True)
    infer_parser.add_argument("--ingredients", type=str, required=True)
    infer_parser.add_argument("--action", type=str, required=True)
    infer_parser.add_argument("--targets", type=str, required=True)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using runtime device: {device}")

    if args.command == "train":
        handle_train(args, device)
    elif args.command == "inference":
        handle_inference(args, device)


if __name__ == "__main__":
    main()
