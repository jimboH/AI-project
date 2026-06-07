#!/usr/bin/env python3
"""Generate semantic IDs for all items using a trained RQ-VAE checkpoint.

Encodes every item's embedding through the frozen RQ-VAE to produce a
discrete code tuple (c0, c1, c2).  Items that share the same 3-code tuple
receive a 4th disambiguation code c3 equal to their collision rank (0-indexed),
matching the convention used by SemanticIdTokenizer.precompute_corpus_ids.

Modality handling
-----------------
Standard (full-modality) run — use --embeddings:
  The provided tensor is used as-is (typically text_embeddings.pt).

Limited-modality run — use --text_embeddings, --image_embeddings, and
  --modality_mask_path together:
  A per-item composite embedding is built:
    "both" / "text_only" items  → text_embeddings[i]
    "image_only" items          → image_embeddings[i]
  This ensures image_only items get semantic IDs derived from their image
  embedding (their text embedding row is zeroed out in the limited tensors).

Outputs
-------
- <save_dir>/semantic_ids.json        : {asin: [c0, c1, c2, c3]}
- <save_dir>/semantic_id_tokens.json  : {asin: "<|d0_X|> <|d1_Y|> <|d2_Z|>"}

Then optionally rewrites every JSONL file under data/<data_dir>/ with
updated `doc_id` and `item` fields.

Usage
-----
  # Standard (full-modality)
  python3 encode_semantic_ids.py \\
      --checkpoint out/rqvae/All_Beauty/cross_modal/checkpoint_best.pt \\
      --embeddings outputs/embeddings/All_Beauty/text_embeddings.pt \\
      --asins      outputs/embeddings/All_Beauty/asins.json \\
      --data_dir   data/amazon_all_beauty \\
      --device     cuda

  # Limited-modality
  python3 encode_semantic_ids.py \\
      --checkpoint        out/rqvae/All_Beauty/cross_modal_limited/checkpoint_best.pt \\
      --text_embeddings   outputs/embeddings/All_Beauty/limited_text_embeddings.pt \\
      --image_embeddings  outputs/embeddings/All_Beauty/limited_image_embeddings.pt \\
      --asins             outputs/embeddings/All_Beauty/asins.json \\
      --modality_mask_path data/limited_modality_mask.json \\
      --out_dir           out/rqvae/All_Beauty/cross_modal_limited/ \\
      --device            cuda
"""

import argparse
import json
import os
import torch

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from modules.rqvae import RqVae


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def codes_to_token_str(codes: list[int]) -> str:
    """Convert a list of integer codes to the token-string format.

    Example: [115, 97, 102] -> '<|d0_115|> <|d1_97|> <|d2_102|>'
    """
    return " ".join(f"<|d{i}_{c}|>" for i, c in enumerate(codes))


def load_model(checkpoint_path: str, device: torch.device) -> RqVae:
    """Load a trained RQ-VAE from a checkpoint file."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = state["model_config"]

    model = RqVae(
        input_dim=cfg["input_dim"],
        embed_dim=cfg["embed_dim"],
        hidden_dims=cfg["hidden_dims"],
        codebook_size=cfg["codebook_size"],
        codebook_kmeans_init=False,
        codebook_normalize=cfg.get("codebook_normalize", False),
        codebook_sim_vq=cfg.get("codebook_sim_vq", False),
        codebook_mode=cfg.get("codebook_mode"),
        codebook_distance_l2_normalize=cfg.get("codebook_distance_l2_normalize", False),
        codebook_use_ema=cfg.get("codebook_use_ema", False),
        codebook_ema_decay=cfg.get("codebook_ema_decay", 0.99),
        codebook_ema_threshold=cfg.get("codebook_ema_threshold", 1.0),
        n_layers=cfg["n_layers"],
        n_cat_features=cfg.get("n_cat_features", 0),
        commitment_weight=cfg.get("commitment_weight", 0.25),
    )

    # strip accelerator prefix if present
    raw = state["model"]
    fixed = {k.replace("module.", "", 1): v for k, v in raw.items()}
    model.load_state_dict(fixed, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded RQ-VAE from {checkpoint_path}  (iter={state['iter']})")
    return model


@torch.no_grad()
def encode_all(
    model: RqVae,
    embeddings: torch.Tensor,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Encode all embeddings and return raw codes: (N, n_layers)."""
    all_codes = []
    n = len(embeddings)
    for start in tqdm(range(0, n, batch_size), desc="Encoding items"):
        batch = embeddings[start : start + batch_size].to(device)
        out = model.get_semantic_ids(batch, gumbel_t=0.0)
        # sem_ids shape: (B, n_layers)
        all_codes.append(out.sem_ids.cpu())
    return torch.cat(all_codes, dim=0)  # (N, n_layers)


def assign_disambiguation(codes: torch.Tensor) -> list[list[int]]:
    """Add a 4th disambiguation code for items sharing the same 3-code tuple.

    Items with a unique (c0, c1, c2) triple get dedup_code = 0 and no 4th token
    is needed (we still store [c0, c1, c2, 0] internally but only emit 3 tokens).
    Items that collide with an earlier item get dedup_code = 1, 2, … in
    encounter order.

    Returns a list of integer lists, one per item.
    """
    n_layers = codes.shape[1]
    seen: dict[tuple, int] = defaultdict(int)  # tuple -> count so far
    result = []
    for row in codes.tolist():
        key = tuple(row[:n_layers])
        dedup = seen[key]
        seen[key] += 1
        result.append(list(row) + [dedup])  # always append the dedup counter
    return result


def build_token_str(full_codes: list[int]) -> str:
    """Build token string; omit d3 token when dedup code is 0."""
    base = codes_to_token_str(full_codes[:3])
    if full_codes[3] != 0:
        base += f" <|d3_{full_codes[3]}|>"
    return base


# ---------------------------------------------------------------------------
# JSONL update
# ---------------------------------------------------------------------------

def update_jsonl(filepath: str, asin_to_token: dict[str, str], dry_run: bool = False) -> int:
    """Rewrite a JSONL file, updating doc_id and item fields where the row has an asin."""
    path = Path(filepath)
    if not path.exists():
        print(f"  [skip] {filepath} — not found")
        return 0

    updated = 0
    new_lines = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                new_lines.append(line)
                continue
            obj = json.loads(line)
            asin = obj.get("asin")
            if asin and asin in asin_to_token:
                token_str = asin_to_token[asin]
                obj["doc_id"] = token_str
                obj["item"]   = token_str
                updated += 1
            new_lines.append(json.dumps(obj, ensure_ascii=False))

    if not dry_run:
        with open(path, "w") as f:
            f.write("\n".join(new_lines) + "\n")

    return updated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_composite_embeddings(
    text_emb: torch.Tensor,
    image_emb: torch.Tensor,
    asins: list,
    modality_mask: dict,
) -> torch.Tensor:
    """Build a per-item composite embedding for the limited-modality case.

    Selection rule (consistent with apply_modality_mask_to_embeddings.py):
      "both" / "text_only" items  → text_emb[i]   (text is present)
      "image_only" items          → image_emb[i]   (text is zeroed out)

    As a safety fallback, any item not found in the mask whose text row is
    zero-norm also falls back to the image embedding.
    """
    composite = text_emb.clone()
    asin_to_idx = {asin: i for i, asin in enumerate(asins)}

    n_image_used = 0
    for asin, status in modality_mask.items():
        idx = asin_to_idx.get(asin)
        if idx is None:
            continue
        if status == "image_only":
            composite[idx] = image_emb[idx]
            n_image_used += 1

    # Fallback: any row still zero after mask lookup → use image embedding
    zero_rows = composite.norm(dim=-1) < 1e-6
    composite[zero_rows] = image_emb[zero_rows]
    n_zero_fallback = zero_rows.sum().item() - n_image_used
    if n_zero_fallback > 0:
        print(f"  [composite] {n_zero_fallback} additional zero-text rows fell back to image embedding")

    print(f"  [composite] image embedding used for {n_image_used:,} image_only items")
    return composite


def main():
    parser = argparse.ArgumentParser(description="Generate RQ-VAE semantic IDs for all items.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint_best.pt")

    # --- Embedding inputs (mutually exclusive modes) ---
    emb_group = parser.add_argument_group(
        "Embedding inputs",
        "Use --embeddings for a standard single-tensor run, OR use "
        "--text_embeddings + --image_embeddings + --modality_mask_path "
        "for the limited-modality composite run."
    )
    emb_group.add_argument("--embeddings", default=None,
                           help="Path to a single embedding tensor (N, D). "
                                "Used for standard (full-modality) runs.")
    emb_group.add_argument("--text_embeddings", default=None,
                           help="Path to limited_text_embeddings.pt (N, D). "
                                "Required for limited-modality runs.")
    emb_group.add_argument("--image_embeddings", default=None,
                           help="Path to limited_image_embeddings.pt (N, D). "
                                "Required for limited-modality runs.")
    emb_group.add_argument("--modality_mask_path", default=None,
                           help="Path to limited_modality_mask.json. "
                                "When provided, a per-item composite embedding is built: "
                                "image_only items use image_embeddings[i], "
                                "all others use text_embeddings[i].")

    parser.add_argument("--asins", required=True,
                        help="Path to asins.json — ordered list of parent_asin strings")
    parser.add_argument("--data_dir", default=None,
                        help="Directory containing JSONL files to update (e.g. data/amazon_all_beauty). "
                             "If omitted, only the JSON mapping files are written.")
    parser.add_argument("--out_dir", default=None,
                        help="Directory for JSON output files. "
                             "Defaults to the same directory as the checkpoint.")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry_run", action="store_true",
                        help="Parse and compute IDs but do not write any files.")
    args = parser.parse_args()

    # Validate embedding argument combinations
    if args.embeddings is not None and (
        args.text_embeddings is not None or args.image_embeddings is not None
    ):
        parser.error("--embeddings cannot be combined with --text_embeddings / --image_embeddings")
    if args.embeddings is None and args.text_embeddings is None:
        parser.error("Provide either --embeddings (standard) or "
                     "--text_embeddings + --image_embeddings (limited-modality)")
    limited_mode = args.modality_mask_path is not None
    if limited_mode and (args.text_embeddings is None or args.image_embeddings is None):
        parser.error("--modality_mask_path requires both --text_embeddings and --image_embeddings")

    device = torch.device(args.device)

    # -- Load model --
    model = load_model(args.checkpoint, device)

    # -- Load embeddings & ASINs --
    with open(args.asins) as f:
        asins = json.load(f)

    if limited_mode:
        print("Limited-modality mode: building composite embeddings ...")
        text_emb  = torch.load(args.text_embeddings,  map_location="cpu", weights_only=False).float()
        image_emb = torch.load(args.image_embeddings, map_location="cpu", weights_only=False).float()
        assert len(asins) == len(text_emb) == len(image_emb), (
            f"ASIN count ({len(asins)}) must match text ({len(text_emb)}) "
            f"and image ({len(image_emb)}) embedding counts"
        )
        with open(args.modality_mask_path) as f:
            modality_mask = json.load(f)
        print(f"  Loaded modality mask with {len(modality_mask):,} entries")
        embeddings = build_composite_embeddings(text_emb, image_emb, asins, modality_mask)
    else:
        embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=False).float()
        assert len(asins) == len(embeddings), (
            f"ASIN count ({len(asins)}) != embedding count ({len(embeddings)})"
        )

    print(f"Loaded {len(asins):,} items  (embedding dim={embeddings.shape[1]})")

    # -- Encode --
    raw_codes = encode_all(model, embeddings, batch_size=args.batch_size, device=device)
    print(f"Raw codes shape: {raw_codes.shape}")

    # -- Assign disambiguation codes --
    full_codes = assign_disambiguation(raw_codes)

    # -- Statistics --
    n_collisions = sum(1 for c in full_codes if c[3] != 0)
    unique_3 = len({tuple(c[:3]) for c in full_codes})
    print(f"Unique 3-code tuples : {unique_3:,} / {len(full_codes):,} items")
    print(f"Items needing d3 code: {n_collisions:,}")

    # -- Build mappings --
    asin_to_codes  = {asin: codes for asin, codes in zip(asins, full_codes)}
    asin_to_token  = {asin: build_token_str(codes) for asin, codes in asin_to_codes.items()}

    # -- Save JSON outputs --
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        ids_path   = out_dir / "semantic_ids.json"
        token_path = out_dir / "semantic_id_tokens.json"
        with open(ids_path, "w") as f:
            json.dump(asin_to_codes, f)
        with open(token_path, "w") as f:
            json.dump(asin_to_token, f, ensure_ascii=False)
        print(f"Saved semantic_ids.json    -> {ids_path}")
        print(f"Saved semantic_id_tokens.json -> {token_path}")

    # -- Update JSONL files --
    if args.data_dir:
        data_dir = Path(args.data_dir)
        jsonl_files = sorted(data_dir.glob("*.jsonl"))
        print(f"\nUpdating {len(jsonl_files)} JSONL file(s) in {data_dir} ...")
        total_updated = 0
        for jf in jsonl_files:
            n = update_jsonl(str(jf), asin_to_token, dry_run=args.dry_run)
            total_updated += n
            print(f"  {jf.name}: {n:,} rows updated")
        print(f"Total rows updated: {total_updated:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
