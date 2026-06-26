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

    def __init__(self, embedding_layer: nn.Module, embed_dim: int, latent_dim: int):
        super().__init__()
        self.embedding_layer = embedding_layer

        self.element_compressor = nn.Sequential(nn.Linear(embed_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())

        self.joint_projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens shape: [batch_size, num_elements, max_len]
        batch_size, num_elements, max_len = tokens.shape

        # 1. Flatten for uniform embedding evaluations
        flat_tokens = tokens.view(-1, max_len)  # [B * N, L]

        # Look for valid tokens (not equal to pad_token_id 0)
        mask = (flat_tokens != 0).float()
        mask_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        embeddings = self.embedding_layer(flat_tokens)  # [B * N, L, D]
        masked_embeddings = embeddings * mask.unsqueeze(-1)

        # Mean pool individual sub-elements text chunks
        pooled_elements = masked_embeddings.sum(dim=1) / mask_counts  # [B * N, D]
        element_latents = self.element_compressor(pooled_elements)  # [B * N, H]

        # 2. Re-assemble back to structured batch layout
        group_latents = element_latents.view(batch_size, num_elements, -1)  # [B, N, H]

        group_mask = (tokens != 0).any(dim=-1).float()  # [B, N]
        group_counts = group_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]

        masked_group = group_latents * group_mask.unsqueeze(-1)  # [B, N, H]
        # combined_latent = masked_group.sum(dim=1) / group_counts  # [B, H]
        #
        # return self.joint_projection(combined_latent), element_latents
        combined_latent = masked_group.sum(dim=1) / group_counts
        projected = self.joint_projection(combined_latent)
        return nn.functional.layer_norm(projected, projected.shape[1:]), element_latents


class ActionSequenceEncoder(nn.Module):
    """Generates structural sequence memories from multi-step action groups for predictor decoding."""

    def __init__(self, embedding_layer: nn.Module, embed_dim: int, latent_dim: int):
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
        # u_seq = self.compressor(pooled_actions).view(batch_size, num_actions, -1)  # [B, N, H]
        #
        a_mask = (a_tokens == 0).all(dim=-1)
        # return u_seq, a_mask

        u_seq = self.compressor(pooled_actions).view(batch_size, num_actions, -1)
        u_seq = nn.functional.layer_norm(u_seq, u_seq.shape[2:])  # Scale bound sequence dimensions
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

    def forward(self, x_tokens, a_tokens):
        # Get intermediate structures and final pooled representation
        z_t, element_latents = self.ingredient_encoder(x_tokens)
        u_seq, a_mask = self.action_encoder(a_tokens)

        # Predictor step
        pred_embed = self.predictor(z_t, u_seq, a_mask)

        # Return everything needed for the VICReg computation
        return pred_embed, z_t, element_latents, u_seq

    @torch.no_grad()
    def update_target_ema(self, momentum: float = 0.999):
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


class FullVicregLoss(nn.Module):
    def __init__(self, sim_weight=1.0, var_weight=25.0, cov_weight=5.0, hinge_epsilon=1e-4):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.eps = hinge_epsilon

    def forward(self, pred_embed, target_embed, ind_ingr, pooled_ingr, action_seq):
        """
        Args:
            pred_embed:   [B, H]  - Predictor output layer
            target_embed: [B, H]  - Ground truth future state latent
            ind_ingr:     [B, N, H] - Individual ingredient embeddings
            pooled_ingr:  [B, H]  - Fully pooled ingredient representation
            action_seq:   [B, A, H] - Sequential action embeddings
        """
        # 0. Core JEPA Similarity Loss (MSE)
        sim_loss = nn.functional.mse_loss(pred_embed, target_embed)

        # 1 & 2: Calculate Variance Losses (Hinge Loss against target std of 1.0)
        # We flatten spatial dimensions [B, N, H] -> [B * N, H] to compute variance over instances
        var_ind_ingr = self._variance_loss(ind_ingr.view(-1, ind_ingr.size(-1)))
        var_pool_ingr = self._variance_loss(pooled_ingr)
        var_actions = self._variance_loss(action_seq.view(-1, action_seq.size(-1)))

        total_var_loss = (var_ind_ingr + var_pool_ingr + var_actions) / 3.0

        # 3 & 4: Calculate Covariance Losses (Decouple feature correlations)
        cov_ind_ingr = self._covariance_loss(ind_ingr.view(-1, ind_ingr.size(-1)))
        cov_pool_ingr = self._covariance_loss(pooled_ingr)
        cov_actions = self._covariance_loss(action_seq.view(-1, action_seq.size(-1)))

        total_cov_loss = (cov_ind_ingr + cov_pool_ingr + cov_actions) / 3.0

        # Weighted Total
        loss = (self.sim_weight * sim_loss) + (self.var_weight * total_var_loss) + (self.cov_weight * total_cov_loss)

        return loss, {"sim": sim_loss.item(), "var": total_var_loss.item(), "cov": total_cov_loss.item()}

    def _variance_loss(self, x):
        """Forces the standard deviation of each feature across the batch to approach 1.0"""
        std = torch.sqrt(x.var(dim=0) + self.eps)
        return torch.mean(nn.functional.relu(1.0 - std))

    def _covariance_loss(self, x):
        """Penalizes off-diagonal correlations linearly using L1 norm"""
        batch_size = x.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=x.device)

        # Center the features
        x = x - x.mean(dim=0, keepdim=True)
        # Compute Covariance Matrix [H, H]
        cov = (x.T @ x) / (batch_size - 1)
        dim = cov.size(0)
        off_diag_abs = cov.abs().sum() - cov.diagonal().abs().sum()
        return off_diag_abs / dim


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
            train_loss_monitor = [0.0, 0.0, 0.0]

            for batch_idx, (x_tokens, a_tokens, y_tokens) in enumerate(self.train_loader):
                vicreg_criterion = FullVicregLoss()
                self.optimizer.zero_grad()
                x_tokens, a_tokens, y_tokens = (
                    x_tokens.to(self.device),
                    a_tokens.to(self.device),
                    y_tokens.to(self.device),
                )

                # pred_embed, z_t = self.model(x_tokens, a_tokens)
                pred_embed, z_t, element_latents, u_seq = self.model(x_tokens, a_tokens)

                with torch.no_grad():
                    target_embed = self.model.encode_target(y_tokens).detach()

                # total_loss, sim_loss, std_loss, cov_loss, repel_loss = vicreg_loss(pred_embed, true_embed, z_t)
                # Compute the unified 4-point constraint loss
                loss, loss_metrics = vicreg_criterion(
                    pred_embed=pred_embed,
                    target_embed=target_embed,
                    ind_ingr=element_latents,
                    pooled_ingr=z_t,
                    action_seq=u_seq,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                self.model.update_target_ema(momentum=0.999)

                train_loss_monitor[0] += loss_metrics["sim"]
                train_loss_monitor[1] += loss_metrics["var"]
                train_loss_monitor[2] += loss_metrics["cov"]

                self.global_step += 1

                if batch_idx % 100 == 0:
                    print(
                        f"Batch: {batch_idx}/{len(self.train_loader)} |",
                        f"SIM: {loss_metrics['sim']:.4f} |",
                        f"STD: {loss_metrics['var']:.4f} |",
                        f"COV: {loss_metrics['cov']:.4f}",
                    )

            epoch_train_sim = train_loss_monitor[0] / len(self.train_loader)
            epoch_train_std = train_loss_monitor[1] / len(self.train_loader)
            epoch_train_cov = train_loss_monitor[2] / len(self.train_loader)
            epoch_val_sim, epoch_val_std, epoch_val_cov = self.validate()

            print(
                f"Epoch {epoch + 1:02d} |",
                f"Train SIM: {epoch_train_sim:.4f} |",
                f"STD: {epoch_train_std:.4f} |",
                f"COV: {epoch_train_cov:.4f}",
            )
            print(
                f"Epoch {epoch + 1:02d} |",
                f"Val SIM: {epoch_val_sim:.4f} |",
                f"STD: {epoch_val_std:.4f} |",
                f"COV: {epoch_val_cov:.4f}",
            )

            if epoch_val_sim < (self.best_val_sim - self.min_delta):
                self.best_val_sim = epoch_val_sim
                self.patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, "recipe_jepa_model_best.pt"))
                print("Saved new best model!")
            else:
                self.patience_counter += 1
                print(f"Early stopping, patience step: {self.patience_counter}")
            print()

            if self.patience_counter >= self.patience:
                print("[🛑] Early stopping triggered.")
                break
        self.writer.close()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss_monitor = [0.0, 0.0, 0.0]
        for x_tokens, a_tokens, y_tokens in self.val_loader:
            x_tokens, a_tokens, y_tokens = x_tokens.to(self.device), a_tokens.to(self.device), y_tokens.to(self.device)
            vicreg_criterion = FullVicregLoss()
            # pred_embed, z_t = self.model(x_tokens, a_tokens)
            pred_embed, z_t, element_latents, u_seq = self.model(x_tokens, a_tokens)
            target_embed = self.model.encode_target(y_tokens).detach()
            loss, loss_metrics = vicreg_criterion(
                pred_embed=pred_embed,
                target_embed=target_embed,
                ind_ingr=element_latents,
                pooled_ingr=z_t,
                action_seq=u_seq,
            )
            # _, sim_loss, std_loss, cov_loss, rep_loss = vicreg_loss(pred_embed, true_embed, z_t)
            val_loss_monitor[0] += loss_metrics["sim"]
            val_loss_monitor[1] += loss_metrics["var"]
            val_loss_monitor[2] += loss_metrics["cov"]
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
        # 1. Forward through full model to get the prediction vector
        pred_embed, _, _, _ = model(x_tokens, a_tokens)

        # 2. Project prediction onto a unit hypersphere
        pred_embed_norm = nn.functional.normalize(pred_embed, p=2, dim=-1)

        results = []
        for target_str in targets_list:
            # 3. Get raw target embedding, explicitly pulling out the input_ids tensor
            target_tokens = tokenizer(target_str, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            target_embed = model.encode_target(target_tokens)  # Now receives [1, sequence_length] tensor

            # 4. Project target onto the same unit hypersphere
            target_embed_norm = nn.functional.normalize(target_embed, p=2, dim=-1)

            # 5. Compute Normalized MSE
            normalized_mse = torch.mean((pred_embed_norm - target_embed_norm) ** 2).item()
            results.append((target_str, normalized_mse))

    # Rank from closest angle to furthest
    results.sort(key=lambda x: x[1])

    print("\n============================================================")
    print(" EVALUATION WITH L2 UNIT-NORMALIZATION")
    print("============================================================")
    for target_str, score in results:
        print(f"{target_str:<30} | Normalized MSE: {score:.6f}")
    print("============================================================")


def main():
    parser = argparse.ArgumentParser(description="Recipe JEPA Unified CLI Workflow")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Workflow Mode")

    train_parser = subparsers.add_parser("train", help="Run model training loop")
    train_parser.add_argument("--train_dataset", type=str, default="data/recipe_train.parquet")
    train_parser.add_argument("--val_dataset", type=str, default="data/recipe_val.parquet")
    train_parser.add_argument("--output_dir", type=str, default="checkpoints")
    train_parser.add_argument("--log_dir", type=str, default="runs/recipe_jepa_experiment")
    train_parser.add_argument("--batch_size", type=int, default=16)
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
