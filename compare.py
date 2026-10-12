"""Compare review sentiment across brands and build an interactive dashboard.

    python compare.py --input data/rental_reviews.csv --out outputs
    open outputs/dashboard.html

Input columns (same as collect_reviews.py output):
    brand, source, date, rating, reviewer_type, label, text   (brand and text are required)
`label` is your own positive/neutral/negative label and is used to check the model.

All statistics come from analyze.py (one source of truth for themes, tests and
thresholds). This script packages them, plus trend, validation and example
sentences, into results.json and a single self-contained dashboard.html that
works offline (no external scripts or fonts).
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import NEG, POS, THEMES, run_analysis, save_outputs, shared_weak_themes

SNIPPET_CHARS = 180   # example sentences are truncated; keep long review text out of the dashboard
EXAMPLES_PER_CELL = 4  # most negative and most positive sentences kept per brand x theme


def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else round(float(obj), 4)
    if isinstance(obj, (pd.Timestamp, datetime.date)):
        return str(obj)[:10]
    if obj is pd.NaT:
        return None
    return obj


def trend_records(df):
    if "date" not in df or df["date"].notna().sum() == 0:
        return []
    d = df.dropna(subset=["date"]).assign(month=lambda x: x["date"].dt.to_period("M").astype(str))
    g = d.groupby(["brand", "month"])["score"].agg(["mean", "count"]).reset_index()
    return [{"brand": r.brand, "month": r.month, "mean": r.mean, "n": int(r.count)} for r in g.itertuples()]


def example_sentences(asp):
    if not len(asp):
        return []
    rows = []
    for (_, _), g in asp.groupby(["brand", "theme"]):
        pick = pd.concat([g.nsmallest(EXAMPLES_PER_CELL, "score"), g.nlargest(EXAMPLES_PER_CELL, "score")])
        rows.append(pick.drop_duplicates("sentence"))
    ex = pd.concat(rows)
    ex["text"] = ex["sentence"].map(lambda s: s if len(s) <= SNIPPET_CHARS else s[:SNIPPET_CHARS].rstrip() + "…")
    return ex[["brand", "theme", "score", "text"]].to_dict("records")


def build_results(res, args):
    df, tb = res["df"], res["tb"].copy()
    if len(tb):
        tb["low_n"] = tb["mentions"] < args.min_theme_n
    val = dict(res["validation"])
    conf = val.pop("confusion", None)
    if conf is not None:
        val["confusion"] = {"rows": list(conf.index), "cols": list(conf.columns), "matrix": conf.values.tolist()}
    synthetic = bool("source" in df and df["source"].astype(str).str.startswith("synthetic").any())
    return clean_json({
        "meta": {"model": args.model, "n_reviews": len(df), "synthetic": synthetic,
                 "min_n": args.min_n, "min_theme_n": args.min_theme_n,
                 "generated": datetime.date.today().isoformat(),
                 "pos_threshold": POS, "neg_threshold": NEG,
                 "date_range": [str(df["date"].min())[:10], str(df["date"].max())[:10]]
                 if "date" in df and df["date"].notna().any() else None},
        "brands": res["bsum"].to_dict("records"),
        "themes": list(THEMES),
        "matrix": tb.to_dict("records"),
        "pairs": res["h2h"].to_dict("records"),
        "sources": res["counts"].to_dict("records"),
        "trend": trend_records(df),
        "shared_weak": shared_weak_themes(tb) if len(tb) else [],
        "validation": val,
        "sentences": example_sentences(res["asp"]),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--model", choices=["vader", "transformer"], default="vader")
    ap.add_argument("--min-n", type=int, default=30, help="reviews per brand needed before a comparison is trusted")
    ap.add_argument("--min-theme-n", type=int, default=5, help="mentions needed before a theme cell is trusted")
    ap.add_argument("--permutations", type=int, default=5000)
    a = ap.parse_args()

    raw = pd.read_csv(a.input)
    if "brand" not in raw or "text" not in raw:
        raise SystemExit("Input needs 'brand' and 'text' columns.")

    res = run_analysis(raw, a.model, a.min_n, a.permutations)
    out = Path(a.out)
    save_outputs(res, out)

    results = build_results(res, a)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    tpl_path = Path(__file__).parent / "dashboard_template.html"
    if not tpl_path.exists():
        raise SystemExit(f"Missing {tpl_path}. Keep dashboard_template.html next to compare.py.")
    html = tpl_path.read_text().replace("__DATA__", json.dumps(results).replace("</", "<\\/"))
    (out / "dashboard.html").write_text(html)

    pd.set_option("display.width", 200)
    print(res["bsum"].round(2).to_string(index=False), "\n")
    for r in results["pairs"]:
        if r.get("p_holm") is None:
            print(f"{r['brand_a']} vs {r['brand_b']}: {r['verdict']}")
        else:
            print(f"{r['brand_a']} vs {r['brand_b']}: diff {r['mean_diff_a_minus_b']:+.2f} "
                  f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]  Holm p={r['p_holm']:.3f}  "
                  f"delta={r['cliffs_delta']:+.2f}  -> {r['verdict']}")
    print(f"\nWrote {out / 'dashboard.html'}")


if __name__ == "__main__":
    main()