import os
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from jepa_cook.src.loss import FullVicregLoss  # deptry: ignore


class JEPATrainer:
    """Core optimization driver managing optimization schedules and validation tracking."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        config: dict[str, Any],
    ) -> None:
        """Binds internal runtime data elements to isolated configurations.

        Args:
            model: Current model optimization framework target.
            train_loader: Core data source tracking training patterns.
            val_loader: Evaluation data source tracking generalizations.
            optimizer: Optimization engine instance.
            scheduler: Learning rate calibration logic driver.
            device: System architecture environment pointer.
            config: Scoped global configuration parameters dictionary.
        """
        self.model: torch.nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.optimizer: torch.optim.Optimizer = optimizer
        self.scheduler: Any = scheduler
        self.device: torch.device = device

        self.train_cfg: dict[str, Any] = config["train"]
        self.model_cfg: dict[str, Any] = config["model"]
        loss_cfg = config.get("loss", {})

        self.vicreg_criterion: FullVicregLoss = FullVicregLoss(
            sim_weight=loss_cfg.get("sim_weight", 1.0),
            var_weight=loss_cfg.get("var_weight", 25.0),
            cov_weight=loss_cfg.get("cov_weight", 5.0),
            hinge_epsilon=loss_cfg.get("hinge_epsilon", 1e-4),
        )

        self.output_dir: str = self.train_cfg["output_dir"]
        self.writer: SummaryWriter = SummaryWriter(log_dir=self.train_cfg["log_dir"])
        self.global_step: int = 0
        self.patience: int = self.train_cfg["patience"]
        self.min_delta: float = self.train_cfg["min_delta"]
        self.best_val_sim: float = float("inf")
        self.patience_counter: int = 0

    def train(self, epochs: int) -> None:
        """Runs the step tracking epoch pipelines.

        Args:
            epochs: Total full system iterations across dataset samples.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        for epoch in range(epochs):
            self.model.train()
            train_loss_monitor: list[float] = [0.0, 0.0, 0.0]

            for batch_idx, (x_tokens, a_tokens, y_tokens) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x_tokens = x_tokens.to(self.device)
                a_tokens = a_tokens.to(self.device)
                y_tokens = y_tokens.to(self.device)

                pred_embed, z_t, element_latents, u_seq = self.model(x_tokens, a_tokens)

                with torch.no_grad():
                    target_embed = self.model.encode_target(y_tokens).detach()

                loss, loss_metrics = self.vicreg_criterion(
                    pred_embed=pred_embed,
                    target_embed=target_embed,
                    ind_ingr=element_latents,
                    pooled_ingr=z_t,
                    action_seq=u_seq,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.train_cfg["clip_grad_norm"])
                self.optimizer.step()
                self.scheduler.step()

                self.model.update_target_ema(momentum=self.model_cfg["ema_momentum"])

                train_loss_monitor[0] += loss_metrics["sim"]
                train_loss_monitor[1] += loss_metrics["var"]
                train_loss_monitor[2] += loss_metrics["cov"]

                self.global_step += 1

                if batch_idx % 100 == 0:
                    print(
                        f"Batch: {batch_idx}/{len(self.train_loader)} | "
                        f"SIM: {loss_metrics['sim']:.4f} | "
                        f"STD: {loss_metrics['var']:.4f} | "
                        f"COV: {loss_metrics['cov']:.4f}"
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
    def validate(self) -> tuple[float, float, float]:
        """Runs the validation subset evaluations.

        Returns:
            Aggregated mean score metrics: similarity, variance, and covariance.
        """
        self.model.eval()
        val_loss_monitor: list[float] = [0.0, 0.0, 0.0]
        for x_tokens, a_tokens, y_tokens in self.val_loader:
            x_tokens = x_tokens.to(self.device)
            a_tokens = a_tokens.to(self.device)
            y_tokens = y_tokens.to(self.device)

            pred_embed, z_t, element_latents, u_seq = self.model(x_tokens, a_tokens)
            target_embed = self.model.encode_target(y_tokens).detach()

            _, loss_metrics = self.vicreg_criterion(
                pred_embed=pred_embed,
                target_embed=target_embed,
                ind_ingr=element_latents,
                pooled_ingr=z_t,
                action_seq=u_seq,
            )
            val_loss_monitor[0] += loss_metrics["sim"]
            val_loss_monitor[1] += loss_metrics["var"]
            val_loss_monitor[2] += loss_metrics["cov"]

        return tuple(v / len(self.val_loader) for v in val_loss_monitor)  # type: ignore
