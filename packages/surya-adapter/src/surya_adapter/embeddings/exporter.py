from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from surya_adapter.loading.model_loader import load_surya_model


def export_embeddings(
    vendor_root,
    config_path,
    checkpoint_path,
    output_path,
    dataset_factory,
    collate_fn,
    num_samples=None,
    device="cpu",
    example_subdir=None,
    project_config=None,
):
    model, config = load_surya_model(
        vendor_root=vendor_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        example_subdir=example_subdir,
        project_config=project_config,
    )
    dataset = dataset_factory(config)
    indices = list(range(len(dataset))) if num_samples is None else list(range(min(num_samples, len(dataset))))
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, collate_fn=collate_fn)

    timestamps: list[str] = []
    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for batch, metadata in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            emb = model(batch, return_embedding=True)
            emb = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
            ts = np.datetime_as_string(metadata['timestamps_input'], unit='m')[0][0]
            timestamps.append(ts)
            embeddings.append(emb)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, timestamps=np.array(timestamps, dtype=str), embeddings=np.stack(embeddings).astype(np.float32))
    return output_path
