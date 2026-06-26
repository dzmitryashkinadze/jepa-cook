import copy

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """Wrapper module around standard PyTorch Embedding layers."""

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """Initializes vocabulary index space and uniform vector dimensions.

        Args:
            vocab_size: Unique language indices limit bound.
            embed_dim: Continuous target dimension width.
        """
        super().__init__()
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Applies vocabulary lookup arrays over tokens.

        Args:
            tokens: Word index matrices.

        Returns:
            Continuous sequence feature space structures.
        """
        return self.embedding(tokens)


class StructuralGroupEncoder(nn.Module):
    """Encodes a nested 3D tensor of multiple item components into a singular joint latent vector."""

    def __init__(self, embedding_layer: nn.Module, embed_dim: int, latent_dim: int) -> None:
        """Initializes embedding references and feature pooling layouts.

        Args:
            embedding_layer: Module processing shared representation definitions.
            embed_dim: Embedding lookup feature dimension width.
            latent_dim: Shared projection vector width.
        """
        super().__init__()
        self.embedding_layer: nn.Module = embedding_layer
        self.element_compressor: nn.Sequential = nn.Sequential(
            nn.Linear(embed_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU()
        )
        self.joint_projection: nn.Sequential = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Maps batch groups to spatial joint latents.

        Args:
            tokens: Dense layouts of shape [batch_size, num_elements, max_len].

        Returns:
            Normalized shared latent maps and individual element feature lists.
        """
        batch_size, num_elements, max_len = tokens.shape
        flat_tokens = tokens.view(-1, max_len)

        mask = (flat_tokens != 0).float()
        mask_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        embeddings = self.embedding_layer(flat_tokens)
        masked_embeddings = embeddings * mask.unsqueeze(-1)

        pooled_elements = masked_embeddings.sum(dim=1) / mask_counts
        element_latents = self.element_compressor(pooled_elements)

        group_latents = element_latents.view(batch_size, num_elements, -1)
        group_mask = (tokens != 0).any(dim=-1).float()
        group_counts = group_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        masked_group = group_latents * group_mask.unsqueeze(-1)
        combined_latent = masked_group.sum(dim=1) / group_counts
        projected = self.joint_projection(combined_latent)

        return nn.functional.layer_norm(projected, projected.shape[1:]), element_latents


class ActionSequenceEncoder(nn.Module):
    """Generates structural sequence memories from multi-step action groups for predictor decoding."""

    def __init__(self, embedding_layer: nn.Module, embed_dim: int, latent_dim: int) -> None:
        """Initializes network modules parsing sequential context elements.

        Args:
            embedding_layer: Module processing shared representation definitions.
            embed_dim: Linear embedding vector width.
            latent_dim: Bottleneck dimension size for structural predictors.
        """
        super().__init__()
        self.embedding_layer: nn.Module = embedding_layer
        self.compressor: nn.Sequential = nn.Sequential(
            nn.Linear(embed_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU()
        )

    def forward(self, a_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes non-zero action listings sequentially.

        Args:
            a_tokens: Structural index tensors of shape [batch_size, num_actions, max_len].

        Returns:
            Scaled bounds embedding structures and active token boolean arrays.
        """
        batch_size, num_actions, max_len = a_tokens.shape
        flat_actions = a_tokens.view(-1, max_len)
        mask = (flat_actions != 0).float()
        mask_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        embeddings = self.embedding_layer(flat_actions)
        masked_embeddings = embeddings * mask.unsqueeze(-1)

        pooled_actions = masked_embeddings.sum(dim=1) / mask_counts
        a_mask = (a_tokens == 0).all(dim=-1)

        u_seq = self.compressor(pooled_actions).view(batch_size, num_actions, -1)
        u_seq = nn.functional.layer_norm(u_seq, u_seq.shape[2:])
        return u_seq, a_mask


class TitleTargetEncoder(nn.Module):
    """Processes flat 2D title targets directly."""

    def __init__(self, embedding_layer: nn.Module, embed_dim: int, latent_dim: int) -> None:
        """Initializes target sequence mapping parameters.

        Args:
            embedding_layer: Independent target representation module space.
            embed_dim: Linear representation width.
            latent_dim: Final hypersphere target space width.
        """
        super().__init__()
        self.embedding_layer: nn.Module = embedding_layer
        self.encoder: nn.Sequential = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compresses target sequence profiles.

        Args:
            tokens: Sequence vectors of shape [batch_size, max_len].

        Returns:
            Continuous decoupled spatial embedding layouts.
        """
        mask = (tokens != 0).float()
        mask_counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        embeddings = self.embedding_layer(tokens)
        masked_embeddings = embeddings * mask.unsqueeze(-1)
        pooled = masked_embeddings.sum(dim=1) / mask_counts

        return self.encoder(pooled)


class Predictor(nn.Module):
    """Transformer decoder predicting sequence targets conditional on masked memories."""

    def __init__(self, latent_dim: int = 256, nhead: int = 8, num_layers: int = 2) -> None:
        """Initializes standard attention parameters.

        Args:
            latent_dim: Operational bottleneck width.
            nhead: Number of parallel self-attention mechanisms.
            num_layers: Depth stack of multi-head sub-blocks.
        """
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            activation=nn.functional.silu,
            batch_first=True,
        )
        self.transformer_decoder: nn.TransformerDecoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, z_t: torch.Tensor, u_seq: torch.Tensor, a_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Passes context vectors through sequence masks.

        Args:
            z_t: Bottleneck maps of shape [batch_size, latent_dim].
            u_seq: Temporal memories of shape [batch_size, sequences, latent_dim].
            a_mask: Token indices to skip.

        Returns:
            Continuous projected candidate positions.
        """
        tgt = z_t.unsqueeze(1)
        mem_key_padding_mask = (a_mask == 0) if a_mask is not None else None
        out = self.transformer_decoder(tgt=tgt, memory=u_seq, memory_key_padding_mask=mem_key_padding_mask)
        return out.squeeze(1)


class RecipeJEPA(nn.Module):
    """Joint-Embedding Predictive Architecture specialized for structured state modeling."""

    def __init__(
        self, vocab_size: int = 30522, embed_dim: int = 384, latent_dim: int = 256, nhead: int = 8, num_layers: int = 2
    ) -> None:
        """Initializes active online encoders and decoupled target parameter layouts.

        Args:
            vocab_size: Token configuration dictionary threshold.
            embed_dim: Feature map internal scale.
            latent_dim: Latent representation processing bottleneck.
            nhead: Predictor attention head count.
            num_layers: Predictor transformer layer count.
        """
        super().__init__()
        self.embedding: Embedding = Embedding(vocab_size, embed_dim)

        self.ingredient_encoder: StructuralGroupEncoder = StructuralGroupEncoder(self.embedding, embed_dim, latent_dim)
        self.action_encoder: ActionSequenceEncoder = ActionSequenceEncoder(self.embedding, embed_dim, latent_dim)

        self.target_embedding = copy.deepcopy(self.embedding)
        self.target_encoder: TitleTargetEncoder = TitleTargetEncoder(self.target_embedding, embed_dim, latent_dim)

        for param in self.target_embedding.parameters():
            param.requires_grad = False
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor: Predictor = Predictor(latent_dim, nhead=nhead, num_layers=num_layers)
        self.delta_norm: nn.LayerNorm = nn.LayerNorm(latent_dim)
        self.action_gate: nn.Parameter = nn.Parameter(torch.tensor([0.1]))
        self.prediction_norm: nn.LayerNorm = nn.LayerNorm(latent_dim)

    def encode_ingredients(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes raw structural input groups through online pipelines.

        Args:
            tokens: High dimensional input sequences.

        Returns:
            Encoded ingredient vectors.
        """
        return self.ingredient_encoder(tokens)

    def encode_target(self, tokens: torch.Tensor) -> torch.Tensor:
        """Extracts continuous evaluation metrics using stable target networks.

        Args:
            tokens: Continuous label matrices.

        Returns:
            Frozen target sequence projections.
        """
        with torch.no_grad():
            return self.target_encoder(tokens)

    def forward(
        self, x_tokens: torch.Tensor, a_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates forward predictions across structural context maps.

        Args:
            x_tokens: Context sequences.
            a_tokens: Multi-step action sets.

        Returns:
            Tuple containing final predictive projections, joint contexts,
            sub-elements latents, and masked sequence steps.
        """
        z_t, element_latents = self.ingredient_encoder(x_tokens)
        u_seq, a_mask = self.action_encoder(a_tokens)

        pred_embed = self.predictor(z_t, u_seq, a_mask)
        return pred_embed, z_t, element_latents, u_seq

    @torch.no_grad()
    def update_target_ema(self, momentum: float = 0.999) -> None:
        """Applies Exponential Moving Average steps to stabilize the target parameter variables.

        Args:
            momentum: Decoupling track speed coefficient scale.
        """
        for target_param, ingredient_param in zip(
            self.target_encoder.parameters(), self.ingredient_encoder.parameters()
        ):
            if target_param.shape == ingredient_param.shape:
                target_param.data.mul_(momentum).add_(ingredient_param.data, alpha=1.0 - momentum)

        for target_embedding_param, embedding_param in zip(
            self.target_embedding.parameters(), self.embedding.parameters()
        ):
            if target_embedding_param.shape == embedding_param.shape:
                target_embedding_param.data.mul_(momentum).add_(embedding_param.data, alpha=1.0 - momentum)
