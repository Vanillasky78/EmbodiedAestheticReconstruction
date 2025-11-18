#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
auto_value_labels.py

Automatically assign estimated value labels to artworks and
write them into embeddings_meta.csv as:

    - value_score  ∈ [0,1]  (if missing, we create one)
    - price_text   = human-readable string, e.g. "Estimate: $2.3M"

Usage examples:

  # Overwrite local CSV in-place
  python backend/tools/auto_value_labels.py \
      --csv data/local/embeddings_meta.csv \
      --out data/local/embeddings_meta.csv \
      --default-tier 1

  # Overwrite Met CSV in-place
  python backend/tools/auto_value_labels.py \
      --csv data/met/embeddings_meta.csv \
      --out data/met/embeddings_meta.csv \
      --default-tier 2

  # Overwrite AIC CSV in-place (will respect existing price_text)
  python backend/tools/auto_value_labels.py \
      --csv data/aic/embeddings_meta.csv \
      --out data/aic/embeddings_meta.csv \
      --default-tier 2
"""

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Optional


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_float(row: Dict, keys, default: float = 0.0) -> float:
    for k in keys:
        if k in row and row[k] not in ("", None):
            try:
                return float(row[k])
            except Exception:
                continue
    return float(default)


def infer_value_score(row: Dict, default_tier: int = 2) -> float:
    """
    Try to infer a value_score ∈ [0,1] from row's existing fields.
    Priority:
        1) value_score / value_norm
        2) tier (1,2,3)
        3) fallback based on default_tier
    """
    vs = safe_float(row, ["value_score", "value_norm"], default=-1.0)
    if vs >= 0.0:
        return clamp01(vs)

    tier = safe_float(row, ["tier"], default=float(default_tier))

    # simple mapping: tier 1 > tier 2 > tier 3
    if tier <= 1:
        base = 0.8
    elif tier <= 2:
        base = 0.6
    elif tier <= 3:
        base = 0.45
    else:
        base = 0.5

    # add a tiny random jitter so values look more "organic"
    jitter = random.uniform(-0.08, 0.08)
    return clamp01(base + jitter)


def estimate_value_usd(value_score: float, tier: int) -> float:
    """
    Map value_score ∈ [0,1] and tier (1,2,3) to an estimated USD amount.

    Rough ranges (you can tweak):
        tier 1 (iconic / masterpiece):   5M – 50M
        tier 2 (major museum portrait):  500K – 5M
        tier 3 (smaller / study):        50K – 500K
        else:                            100K – 1M

    We use an exponential interpolation so that higher scores
    "explode" more at the high end.
    """
    s = clamp01(value_score)

    if tier == 1:
        low, high = 5_000_000, 50_000_000
    elif tier == 2:
        low, high = 500_000, 5_000_000
    elif tier == 3:
        low, high = 50_000, 500_000
    else:
        low, high = 100_000, 1_000_000

    # exponential interpolation: amount = low * (high/low) ** s
    ratio = high / low
    amount = low * (ratio ** s)
    return float(amount)


def format_price_text(amount_usd: float) -> str:
    """
    Format USD amount into a human-readable "Estimate: $X.YM" style.

    Examples:
        5_200_000 → "Estimate: $5.2M"
        850_000   → "Estimate: $850K"
        75_000    → "Estimate: $75K"
        9_500     → "Estimate: $9.5K"
    """
    n = float(amount_usd)

    # millions
    if n >= 1_000_000:
        val = n / 1_000_000.0
        # 1 decimal if < 10M, otherwise integer
        if val < 10:
            return f"Estimate: ${val:.1f}M"
        else:
            return f"Estimate: ${val:.0f}M"

    # thousands
    if n >= 1_000:
        val = n / 1_000.0
        if val < 10:
            return f"Estimate: ${val:.1f}K"
        else:
            return f"Estimate: ${val:.0f}K"

    # below 1k
    return f"Estimate: ${n:,.0f}"


def process_csv(
    csv_path: Path,
    out_path: Optional[Path] = None,
    default_tier: int = 2,
    preserve_existing_price: bool = True,
) -> None:
    """
    Read embeddings_meta.csv, assign value_score & price_text if missing,
    and write to out_path (or overwrite original if out_path is None).
    """
    if out_path is None:
        out_path = csv_path

    print(f"[INFO] Input CSV:  {csv_path}")
    print(f"[INFO] Output CSV: {out_path}")
    print(f"[INFO] Default tier (fallback): {default_tier}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: List[Dict] = list(reader)
        fieldnames = list(reader.fieldnames or [])

    # ensure required columns exist
    for col in ["value_score", "price_text", "tier"]:
        if col not in fieldnames:
            fieldnames.append(col)

    updated_rows: List[Dict] = []

    for idx, row in enumerate(rows):
        # try to keep existing tier if present
        tier_val = safe_float(row, ["tier"], default=float(default_tier))
        tier_int = int(round(tier_val)) if tier_val > 0 else default_tier
        row["tier"] = str(tier_int)

        # if there is already a price_text and we want to preserve it, keep it
        existing_price = str(row.get("price_text") or "").strip()
        if existing_price and preserve_existing_price:
            # but we may still want a normalized value_score
            vs = infer_value_score(row, default_tier=tier_int)
            row["value_score"] = f"{vs:.3f}"
            updated_rows.append(row)
            continue

        # infer / create value_score
        vs = infer_value_score(row, default_tier=tier_int)
        row["value_score"] = f"{vs:.3f}"

        # map to USD amount & format
        amount = estimate_value_usd(vs, tier=tier_int)
        price_text = format_price_text(amount)
        row["price_text"] = price_text

        updated_rows.append(row)

    # write back
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in updated_rows:
            writer.writerow(r)

    print(f"[OK] Wrote {len(updated_rows)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Input embeddings_meta.csv path.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: overwrite input).",
    )
    parser.add_argument(
        "--default-tier",
        type=int,
        default=2,
        help="Fallback tier if row does not contain 'tier' field.",
    )
    parser.add_argument(
        "--no-preserve-existing-price",
        action="store_true",
        help="If set, overwrite existing price_text as well.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out is not None else None

    process_csv(
        csv_path=csv_path,
        out_path=out_path,
        default_tier=args.default_tier,
        preserve_existing_price=not args.no_preserve_existing_price,
    )


if __name__ == "__main__":
    main()
