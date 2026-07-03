"""Offline training entry point.

Usage:
    python training/train.py --interactions data/interactions.csv --catalog data/catalog.csv
"""
import argparse, logging, sys
from pathlib import Path
import pandas as pd
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.collaborative_filter import CollaborativeFilter
from training.content_based import ContentBasedModel

WEIGHT_MAP = {"click": 1, "add_to_cart": 3, "purchase": 5}

def load_interactions(path):
    df = pd.read_csv(path)
    df["weight"] = df["event"].map(WEIGHT_MAP).fillna(1).astype(float)
    return df.groupby(["user_id","sku"], as_index=False)["weight"].max()

def main(args):
    logger.info("=== Training pipeline start ===")
    interactions = load_interactions(args.interactions)
    catalog = pd.read_csv(args.catalog)
    cf = CollaborativeFilter(method=args.model, n_factors=args.factors).fit(interactions)
    cf.save(args.output)
    cb = ContentBasedModel().fit(catalog)
    cb.save(args.output)
    sample = interactions["user_id"].iloc[0]
    logger.info("Sample recs for %s: %s", sample, [r[0] for r in cf.recommend_for_user(sample, 5)])
    logger.info("=== Done. Artifacts saved to %s ===", args.output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interactions", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--model", default="als", choices=["als","svd"])
    p.add_argument("--factors", type=int, default=100)
    p.add_argument("--output", default="models/")
    main(p.parse_args())
