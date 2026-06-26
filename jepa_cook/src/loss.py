import torch
import torch.nn as nn


class FullVicregLoss(nn.Module):
    """VICReg criterion handling multi-dimensional collapse-prevention constraints."""

    def __init__(
        self, sim_weight: float = 1.0, var_weight: float = 25.0, cov_weight: float = 5.0, hinge_epsilon: float = 1e-4
    ) -> None:
        """Initializes internal criteria tuning scalar coefficients.

        Args:
            sim_weight: Mean squared proximity optimizer multiplier.
            var_weight: Cross-instance variance enforcement threshold weight.
            cov_weight: Independent feature decoupling penalty score weight.
            hinge_epsilon: Protection margin calculation floor boundary.
        """
        super().__init__()
        self.sim_weight: float = sim_weight
        self.var_weight: float = var_weight
        self.cov_weight: float = cov_weight
        self.eps: float = hinge_epsilon

    def forward(
        self,
        pred_embed: torch.Tensor,
        target_embed: torch.Tensor,
        ind_ingr: torch.Tensor,
        pooled_ingr: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes structural variance, covariance, and proximity metrics.

        Args:
            pred_embed: Candidate vectors of shape [B, H].
            target_embed: Ground truth projections of shape [B, H].
            ind_ingr: Element subcomponents of shape [B, N, H].
            pooled_ingr: Context representations of shape [B, H].
            action_seq: Action sequences of shape [B, A, H].

        Returns:
            Weighted aggregate objective variables alongside unscaled structural loss logs.
        """
        sim_loss = nn.functional.mse_loss(pred_embed, target_embed)

        var_ind_ingr = self._variance_loss(ind_ingr.view(-1, ind_ingr.size(-1)))
        var_pool_ingr = self._variance_loss(pooled_ingr)
        var_actions = self._variance_loss(action_seq.view(-1, action_seq.size(-1)))
        total_var_loss = (var_ind_ingr + var_pool_ingr + var_actions) / 3.0

        cov_ind_ingr = self._covariance_loss(ind_ingr.view(-1, ind_ingr.size(-1)))
        cov_pool_ingr = self._covariance_loss(pooled_ingr)
        cov_actions = self._covariance_loss(action_seq.view(-1, action_seq.size(-1)))
        total_cov_loss = (cov_ind_ingr + cov_pool_ingr + cov_actions) / 3.0

        loss = (self.sim_weight * sim_loss) + (self.var_weight * total_var_loss) + (self.cov_weight * total_cov_loss)

        return loss, {"sim": sim_loss.item(), "var": total_var_loss.item(), "cov": total_cov_loss.item()}

    def _variance_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Forces standard deviation profiles to approach standard scale parameters."""
        std = torch.sqrt(x.var(dim=0) + self.eps)
        return torch.mean(nn.functional.relu(1.0 - std))

    def _covariance_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Penalizes multi-feature correlation matrices off-diagonal variables."""
        batch_size = x.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=x.device)

        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)
        dim = cov.size(0)
        off_diag_abs = cov.abs().sum() - cov.diagonal().abs().sum()
        return off_diag_abs / dim
