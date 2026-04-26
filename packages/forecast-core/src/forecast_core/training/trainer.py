from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from forecast_core.data_pipelines.omni_dscovr_sdo.scalers import ForecastScalers
from forecast_core.evaluation.event_metrics import summarize_bz_event_metrics
from forecast_core.evaluation.regression_metrics import summarize_regression_metrics
from forecast_core.models.forecast_model import ForecastModel
from forecast_core.training.callbacks import BestCheckpoint
from forecast_core.training.losses import compute_forecast_loss
from forecast_core.training.schedulers import build_warmup_cosine_scheduler


@dataclass(slots=True)
class ForecastBatch:
    image_embed: torch.Tensor
    image_embed_mask: torch.Tensor
    history: torch.Tensor
    context: torch.Tensor
    persistence: torch.Tensor
    targets: torch.Tensor
    bz_event_targets: torch.Tensor


class ForecastTrainer:
    def __init__(self, config_path: str | Path):
        self.config = yaml.safe_load(Path(config_path).read_text())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ForecastModel(self.config).to(self.device)
        tcfg = self.config['training']
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay'])
        self.scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        self.loss_weights = torch.tensor(tcfg['loss_weights'], device=self.device)
        self.time_weights = torch.linspace(1.0, 1.5, steps=self.config['model']['output_steps'], device=self.device)
        self.best_ckpt = BestCheckpoint(tcfg['checkpoint_path'])
        self.forecast_scalers = ForecastScalers.from_yaml(self.config['data']['scaler_path'])

    def _load_npz(self, path: str | Path) -> TensorDataset:
        data = np.load(path)
        history = torch.tensor(data['history'], dtype=torch.float32)
        context = torch.tensor(data['context'], dtype=torch.float32)
        targets = torch.tensor(data['targets'], dtype=torch.float32)
        persistence = torch.tensor(data['persistence'], dtype=torch.float32)
        image_embed_mask = torch.tensor(data['image_embed_mask'], dtype=torch.float32) if 'image_embed_mask' in data.files else torch.ones((len(history), 1), dtype=torch.float32)
        dataset = TensorDataset(
            torch.tensor(data['image_embed'], dtype=torch.float32),
            image_embed_mask,
            self.forecast_scalers.history.normalize(history),
            self.forecast_scalers.context.normalize(context),
            self.forecast_scalers.target.normalize(persistence),
            self.forecast_scalers.target.normalize(targets),
            torch.tensor(data['bz_event_targets'], dtype=torch.float32),
        )
        return dataset

    def build_loaders(self) -> tuple[DataLoader, DataLoader]:
        tcfg = self.config['training']
        train_ds = self._load_npz(self.config['data']['train_npz'])
        valid_ds = self._load_npz(self.config['data']['valid_npz'])
        train_loader = DataLoader(train_ds, batch_size=tcfg['batch_size'], shuffle=True, num_workers=tcfg.get('num_workers', 0), drop_last=True)
        valid_loader = DataLoader(valid_ds, batch_size=tcfg['batch_size'], shuffle=False, num_workers=tcfg.get('num_workers', 0))
        return train_loader, valid_loader

    def _move(self, batch) -> ForecastBatch:
        return ForecastBatch(*(x.to(self.device) for x in batch))

    def fit(self) -> None:
        train_loader, valid_loader = self.build_loaders()
        total_steps = len(train_loader) * self.config['training']['max_epochs']
        warmup_steps = len(train_loader) * self.config['training'].get('warmup_epochs', 1)
        scheduler = build_warmup_cosine_scheduler(self.optimizer, warmup_steps, total_steps)
        for epoch in range(self.config['training']['max_epochs']):
            self.model.train()
            for batch in train_loader:
                batch = self._move(batch)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(batch.image_embed, batch.history, batch.context, batch.persistence, batch.image_embed_mask)
                    loss, _ = compute_forecast_loss(outputs, batch.targets, batch.bz_event_targets, self.loss_weights, self.time_weights)
                self.scaler_amp.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                self.scaler_amp.step(self.optimizer)
                self.scaler_amp.update()
                scheduler.step()
            metrics = self.evaluate(valid_loader)
            payload = {'model': self.model.state_dict(), 'config': self.config, 'epoch': epoch, 'metrics': metrics}
            self.best_ckpt.update(metrics['aggregate']['val_loss'], payload)
            print(f"epoch={epoch+1} val_loss={metrics['aggregate']['val_loss']:.4f}")

    def evaluate(self, valid_loader: DataLoader) -> dict:
        self.model.eval()
        val_losses = []
        preds = []
        targets = []
        event_logits = []
        event_targets = []
        with torch.no_grad():
            for batch in valid_loader:
                batch = self._move(batch)
                outputs = self.model(batch.image_embed, batch.history, batch.context, batch.persistence, batch.image_embed_mask)
                loss, _ = compute_forecast_loss(outputs, batch.targets, batch.bz_event_targets, self.loss_weights, self.time_weights)
                val_losses.append(float(loss.detach().cpu()))
                preds.append(self.forecast_scalers.target.denormalize(outputs['mean']).cpu().numpy())
                targets.append(self.forecast_scalers.target.denormalize(batch.targets).cpu().numpy())
                event_logits.append(outputs['bz_event_logits'].cpu().numpy())
                event_targets.append(batch.bz_event_targets.cpu().numpy())
        pred_np = np.concatenate(preds, axis=0)
        target_np = np.concatenate(targets, axis=0)
        logits_np = np.concatenate(event_logits, axis=0)
        event_true_np = np.concatenate(event_targets, axis=0)
        return {
            'aggregate': {'val_loss': float(np.mean(val_losses))},
            'regression': summarize_regression_metrics(target_np, pred_np),
            'events': summarize_bz_event_metrics(event_true_np, logits_np),
        }
