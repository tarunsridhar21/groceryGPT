"""Fetches UK grocery product data from Open Food Facts and saves to parquet."""
import argparse
import time
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from src.config import DATA_DIR, PRODUCT_LIMIT

_API_URL = (
    "https://world.openfoodfacts.org/api/v2/search"
    "?countries_tags=united-kingdom"
    "&fields=code,product_name,brands,categories,ingredients_text,"
    "nutriments,quantity,nutriscore_grade,nova_group,allergens,labels,stores"
    "&page_size=100"
    "&page={page}"
)

_HEADERS = {"User-Agent": "GroceryGPT/1.0 (educational)"}


def _build_text(row: pd.Series) -> str:
    return (
        f"Product: {row.get('product_name', '')}\n"
        f"Brand: {row.get('brands', '')}\n"
        f"Categories: {row.get('categories', '')}\n"
        f"Ingredients: {row.get('ingredients_text', '')}\n"
        f"Allergens: {row.get('allergens', '')}\n"
        f"Labels: {row.get('labels', '')}\n"
        f"Nutri-Score: {row.get('nutriscore_grade', '')}\n"
        f"NOVA group: {row.get('nova_group', '')}\n"
        f"Quantity: {row.get('quantity', '')}"
    )


def _fetch_page(url: str, max_retries: int = 4) -> list[dict[str, Any]]:
    """Fetch a single API page with exponential back-off on 5xx errors."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=_HEADERS, timeout=30)
            if response.status_code in (429, 500, 502, 503, 504):
                wait = delay * (2 ** attempt)
                print(f"HTTP {response.status_code} on attempt {attempt + 1}; retrying in {wait:.0f}s ...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json().get("products", [])
        except requests.RequestException as exc:
            wait = delay * (2 ** attempt)
            print(f"Request error on attempt {attempt + 1}: {exc}; retrying in {wait:.0f}s ...")
            time.sleep(wait)
    print(f"Giving up after {max_retries} retries for {url}")
    return []


def fetch_uk_products(limit: int = PRODUCT_LIMIT) -> pd.DataFrame:
    """Fetch up to `limit` UK products from Open Food Facts v2 API."""
    records: list[dict[str, Any]] = []
    page = 1
    pbar = tqdm(total=limit, desc="Fetching products", unit="products")

    while len(records) < limit:
        url = _API_URL.format(page=page)
        products = _fetch_page(url)
        if not products:
            break

        records.extend(products)
        fetched = min(len(records), limit)
        pbar.n = fetched
        pbar.refresh()
        page += 1
        time.sleep(0.3)

    pbar.close()

    df = pd.DataFrame(records[:limit])

    # Normalise expected columns; fill missing with empty string
    for col in [
        "code", "product_name", "brands", "categories",
        "ingredients_text", "quantity", "nutriscore_grade",
        "nova_group", "allergens", "labels",
    ]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # Drop rows missing essential fields
    df = df[df["product_name"].str.strip() != ""]
    df = df[df["ingredients_text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["code"])
    df = df.reset_index(drop=True)

    df["text"] = df.apply(_build_text, axis=1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "products.parquet"

    # Merge with any previously fetched products so reruns accumulate data
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        merged = pd.concat([existing, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["code"]).reset_index(drop=True)
        if len(merged) > len(df):
            print(f"Merged {len(df)} new products with {len(existing)} existing -> {len(merged)} total")
            df = merged

    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} products to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest UK grocery products from Open Food Facts.")
    parser.add_argument("--limit", type=int, default=PRODUCT_LIMIT, help="Max products to fetch")
    args = parser.parse_args()
    fetch_uk_products(limit=args.limit)
