#!/usr/bin/env python3
"""Collect public reviews for Nuuly, BNTO and Rent the Runway from sources
that provide official, documented access (no HTML scraping of sites whose
terms forbid it).

Sources
  appstore  Apple's public customer-reviews RSS feed + iTunes Search API
  reddit    Reddit's official API via PRAW (needs free credentials)
  csv       Normalise manual or official exports (e.g. a Trustpilot business
            export, or reviews you copied by hand) into the same schema

Output columns (matches the project template):
  brand,source,date,rating,reviewer_type,label,text

Usage
  pip install requests pandas praw
  python collect_reviews.py appstore --out data/appstore.csv
  REDDIT_CLIENT_ID=... REDDIT_CLIENT_SECRET=... REDDIT_USER_AGENT="review-sentiment by u/you" \
      python collect_reviews.py reddit --out data/reddit.csv
  python collect_reviews.py csv --input my_export.csv --brand BNTO --source trustpilot-export \
      --text-col "Review Content" --rating-col Stars --date-col "Date" --out data/manual.csv
  python collect_reviews.py merge data/*.csv --out data/rental_reviews.csv

Before using any source, read its terms. Keep only review text, date and
rating; never store usernames or other personal details.
"""
import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime

import requests

BRANDS = {
    "Nuuly": {"appstore_term": "Nuuly", "reddit_query": '"Nuuly"'},
    "BNTO": {"appstore_term": "BNTO", "reddit_query": '"BNTO" (rental OR clothing OR subscription)'},
    "Rent the Runway": {"appstore_term": "Rent the Runway", "reddit_query": '"Rent the Runway" OR "RTR"'},
}
# Strings that must appear in the App Store listing name to avoid wrong matches.
APP_NAME_MATCH = {"Nuuly": "nuuly", "BNTO": "bnto", "Rent the Runway": "rent the runway"}
FIELDS = ["brand", "source", "date", "rating", "reviewer_type", "label", "text"]
HEADERS = {"User-Agent": "review-sentiment-research/1.0 (portfolio project)"}
DELAY = 1.5  # seconds between requests; be polite


def clean(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def write_rows(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    seen, out = set(), []
    for r in rows:  # dedupe within brand, drop blanks
        key = (r["brand"], r["text"].lower())
        if not r["text"] or key in seen:
            continue
        seen.add(key)
        out.append(r)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    counts = {}
    for r in out:
        counts[r["brand"]] = counts.get(r["brand"], 0) + 1
    print(f"Wrote {len(out)} rows to {path}: {counts}")
    for b, n in counts.items():
        if n < 30:
            print(f"  note: {b} has {n} reviews, below the 30-review minimum")


# ---------------------------------------------------------------- App Store
def find_app_id(brand):
    r = requests.get(
        "https://itunes.apple.com/search",
        params={"term": BRANDS[brand]["appstore_term"], "entity": "software", "country": "us", "limit": 5},
        headers=HEADERS, timeout=30,
    )
    r.raise_for_status()
    for app in r.json().get("results", []):
        if APP_NAME_MATCH[brand] in app.get("trackName", "").lower():
            return app["trackId"], app["trackName"]
    return None, None


def appstore(args):
    rows = []
    overrides = dict(kv.split("=", 1) for kv in (args.app_id or []))
    for brand in BRANDS:
        app_id = overrides.get(brand)
        if not app_id:
            app_id, name = find_app_id(brand)
            print(f"{brand}: matched app {name} ({app_id})" if app_id else f"{brand}: no App Store app found")
        if not app_id:
            continue
        for page in range(1, 11):  # Apple exposes up to 10 pages (~500 reviews)
            url = f"https://itunes.apple.com/us/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                break
            entries = resp.json().get("feed", {}).get("entry", [])
            if isinstance(entries, dict):
                entries = [entries]
            got = 0
            for e in entries:
                if "im:rating" not in e:  # first entry can be app metadata
                    continue
                text = clean(f'{e["title"]["label"]}. {e["content"]["label"]}')
                rows.append({
                    "brand": brand, "source": "apple-app-store",
                    "date": e["updated"]["label"][:10],
                    "rating": int(e["im:rating"]["label"]),
                    "reviewer_type": "consumer", "label": "", "text": text,
                })
                got += 1
            if got == 0:
                break
            time.sleep(DELAY)
    write_rows(rows, args.out)


# ------------------------------------------------------------------- Reddit
def reddit(args):
    try:
        import praw
    except ImportError:
        sys.exit("pip install praw")
    for v in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"):
        if not os.environ.get(v):
            sys.exit(f"Set {v} (create a script app at reddit.com/prefs/apps)")
    client = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )
    rows = []
    for brand, cfg in BRANDS.items():
        for post in client.subreddit(args.subreddits).search(cfg["reddit_query"], sort="new", limit=args.limit):
            text = clean(f"{post.title}. {post.selftext}")
            if APP_NAME_MATCH[brand] not in text.lower() and brand != "BNTO":
                continue
            rows.append({
                "brand": brand, "source": "reddit",
                "date": datetime.utcfromtimestamp(post.created_utc).strftime("%Y-%m-%d"),
                "rating": "", "reviewer_type": "consumer", "label": "", "text": text,
            })
            time.sleep(0.2)
    write_rows(rows, args.out)


# ---------------------------------------------------------------------- CSV
def csv_import(args):
    import pandas as pd
    df = pd.read_csv(args.input)
    rows = []
    for _, r in df.iterrows():
        date = ""
        if args.date_col and str(r.get(args.date_col, "")) != "nan":
            date = pd.to_datetime(r[args.date_col], errors="coerce")
            date = "" if pd.isna(date) else date.strftime("%Y-%m-%d")
        rating = r.get(args.rating_col, "") if args.rating_col else ""
        rows.append({
            "brand": args.brand, "source": args.source, "date": date,
            "rating": "" if str(rating) == "nan" else int(float(rating)) if str(rating) else "",
            "reviewer_type": "consumer", "label": "", "text": clean(str(r[args.text_col])),
        })
    write_rows(rows, args.out)


def merge(args):
    rows = []
    for p in args.files:
        with open(p, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    write_rows(rows, args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("appstore")
    a.add_argument("--app-id", nargs="*", help='override, e.g. "Nuuly=123456789"')
    a.add_argument("--out", default="data/appstore.csv")
    a.set_defaults(fn=appstore)

    r = sub.add_parser("reddit")
    r.add_argument("--subreddits", default="all")
    r.add_argument("--limit", type=int, default=100)
    r.add_argument("--out", default="data/reddit.csv")
    r.set_defaults(fn=reddit)

    c = sub.add_parser("csv")
    c.add_argument("--input", required=True)
    c.add_argument("--brand", required=True, choices=list(BRANDS))
    c.add_argument("--source", required=True)
    c.add_argument("--text-col", required=True)
    c.add_argument("--rating-col")
    c.add_argument("--date-col")
    c.add_argument("--out", default="data/manual.csv")
    c.set_defaults(fn=csv_import)

    m = sub.add_parser("merge")
    m.add_argument("files", nargs="+")
    m.add_argument("--out", default="data/rental_reviews.csv")
    m.set_defaults(fn=merge)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()