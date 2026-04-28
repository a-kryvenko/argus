# Surya integration boundary

## Vendor repo
Place the official NASA repo at:

- `vendor/surya/`

This directory is treated as a pinned upstream dependency.

## Argus-owned glue
Use `packages/surya-adapter/` for:

- model loading
- embedding export
- embedding store and alignment
- SDO manifest handling
- optional image forecast wrappers

## Main flow
1. Sync SDO data into `data/raw/sdo/`
2. Build an SDO manifest
3. Export pooled Surya embeddings into `data/features/surya/surya_embeddings.npz`
4. Feed those embeddings into `forecast-core`
