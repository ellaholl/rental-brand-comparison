"""Sentiment and theme analysis for public reviews, with brand comparison.

Works with the output of collect_reviews.py (data/rental_reviews.csv).

Usage:
    python analyze.py --input data/rental_reviews.csv --out outputs
    python analyze.py --input data/rental_reviews.csv --model transformer --min-n 30

Input CSV columns:
    brand           optional, e.g. Nuuly / BNTO / Rent the Runway.
                    If missing, all rows are treated as one brand.
    text            required, the review text
    source          optional, e.g. "apple-app-store", "reddit"
    date            optional, YYYY-MM-DD
    rating          optional, 1-5 stars (blank is fine, e.g. Reddit posts)
    reviewer_type   optional, "consumer" or "owner"
    label           optional, your hand label: positive / neutral / negative

Outputs (in --out):
    reviews_scored.csv, aspect_sentences.csv, source_counts.csv,
    brand_summary.csv, theme_by_brand.csv, head_to_head.csv,
    sentiment_by_brand.png, sentiment_by_theme.png, sentiment_over_time.png
"""
import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

THEMES = {
    "fit_sizing":    r"\bfit\b|\bfits\b|\bfitting|\bsiz(e|es|ed|ing)\b|too (small|big|tight|loose|short|long)|runs (small|large)|tailor|alteration",
    "condition":     r"dirty|stain|smell|odor|wrinkl|\btorn|\brip|\bhole|missing|damag|\bworn|quality|condition|\bclean|fabric|button|zipper",
    "delivery":      r"\bship|deliver|arriv|\blate\b|\breturn|prepaid|\blabel|\bbox\b|package|tracking|fedex|\bups\b|usps|on time",
    "selection":     r"selection|\bstyles?\b|variety|\boptions?\b|\bchoice|designer|inventory|catalog|trend|\bnew arrivals",
    "pricing":       r"\bpric(e|es|ed|ing)\b|\bcost|expensive|cheap|afford|\bvalue\b|worth|\$|overpriced|\bdeal\b",
    "subscription":  r"subscri|member|cancel|\bpause|\bskip|renew|\bcharge|billing|\bbill\b|refund|\bplan\b|credit|late fee",
    "service":       r"customer service|support|representative|\bagent|respond|\bemail|\bchat\b|helpful|\brude|resolv|\bcompensat",
    "app_website":   r"\bapp\b|website|\bsite\b|log ?in|crash|glitch|\bbug|search|filter|checkout|interface|\bui\b",
}
POS, NEG = 0.05, -0.05  # standard VADER thresholds
MIN_N_DEFAULT = 30
MIN_THEME_MENTIONS = 5  # theme cells with fewer mentions are flagged low_n


# ------------------------------------------------------------------ cleaning
def clean(df):
    df = df.copy()
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[(df["text"].str.len() > 0) & (df["text"].str.lower() != "nan")]
    if "brand" not in df:
        df["brand"] = "All"
    df["brand"] = df["brand"].fillna("Unknown").astype(str).str.strip()
    if "source" not in df:
        df["source"] = "unknown"
    df["source"] = df["source"].fillna("unknown").astype(str)
    if "rating" in df:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "label" in df:
        df["label"] = df["label"].astype(str).str.strip().str.lower().replace({"nan": np.nan, "": np.nan})
    # Duplicates are removed within each brand only: the same text under two
    # brands is not a duplicate.
    key = df["brand"] + "|" + df["text"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    df = df[~key.duplicated()].reset_index(drop=True)
    df["review_id"] = df.index
    return df


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def tag_themes(sentence, themes=None):
    s = sentence.lower()
    return [name for name, pat in (themes or THEMES).items() if re.search(pat, s)]


def label_from_score(score):
    return "positive" if score >= POS else "negative" if score <= NEG else "neutral"


# ------------------------------------------------------------------- scoring
def make_scorer(kind):
    if kind == "vader":
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return lambda texts: [sia.polarity_scores(t)["compound"] for t in texts]
    from transformers import pipeline  # optional heavy dependency
    clf = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None)

    def score(texts):
        out = []
        for res in clf(list(texts), truncation=True):
            p = {r["label"].lower(): r["score"] for r in res}
            out.append(p.get("positive", 0) - p.get("negative", 0))
        return out
    return score


# ---------------------------------------------------------------- statistics
def bootstrap_ci(values, n=1000, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def bootstrap_diff_ci(a, b, n=2000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ma = rng.choice(a, size=(n, len(a)), replace=True).mean(axis=1)
    mb = rng.choice(b, size=(n, len(b)), replace=True).mean(axis=1)
    return tuple(np.percentile(ma - mb, [2.5, 97.5]))


def permutation_p(a, b, n=5000, seed=0):
    """Two-sided permutation test on the difference in means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    count = 0
    for _ in range(n):
        rng.shuffle(pooled)
        if abs(pooled[:len(a)].mean() - pooled[len(a):].mean()) >= observed:
            count += 1
    return (count + 1) / (n + 1)


def cliffs_delta(a, b):
    """Cliff's delta via the Mann-Whitney U rank formula (O(n log n))."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ranks = pd.Series(np.concatenate([a, b])).rank().to_numpy()
    u = ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return 2 * u / (len(a) * len(b)) - 1


def effect_label(d):
    d = abs(d)
    return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def spearman(x, y):
    return pd.Series(x).rank().corr(pd.Series(y).rank())


# ----------------------------------------------------------------- summaries
def source_counts(df):
    return df.groupby(["brand", "source"]).size().rename("reviews").reset_index()


def brand_summary(df, min_n):
    rows = []
    for brand, g in df.groupby("brand"):
        lo, hi = bootstrap_ci(g["score"])
        pos = (g["pred"] == "positive").mean() * 100
        neg = (g["pred"] == "negative").mean() * 100
        row = {"brand": brand, "reviews": len(g), "mean_sentiment": g["score"].mean(),
               "ci_low": lo, "ci_high": hi,
               "pct_positive": pos, "pct_neutral": 100 - pos - neg, "pct_negative": neg,
               "net_sentiment": pos - neg,
               "status": "ok" if len(g) >= min_n else f"below minimum ({min_n})"}
        if "rating" in g and g["rating"].notna().sum() > 3:
            d = g.dropna(subset=["rating"])
            row["avg_rating"] = d["rating"].mean()
            row["rated_reviews"] = len(d)
            row["spearman_vs_rating"] = spearman(d["score"], d["rating"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_sentiment", ascending=False).reset_index(drop=True)


def head_to_head(df, min_n, n_perm=5000):
    rows, testable = [], []
    for a, b in combinations(sorted(df["brand"].unique()), 2):
        sa = df.loc[df["brand"] == a, "score"].to_numpy()
        sb = df.loc[df["brand"] == b, "score"].to_numpy()
        row = {"brand_a": a, "brand_b": b, "n_a": len(sa), "n_b": len(sb)}
        if min(len(sa), len(sb)) < min_n:
            row["verdict"] = f"no verdict (needs {min_n}+ reviews per brand)"
        else:
            lo, hi = bootstrap_diff_ci(sa, sb)
            d = cliffs_delta(sa, sb)
            row.update({"mean_diff_a_minus_b": sa.mean() - sb.mean(), "ci_low": lo, "ci_high": hi,
                        "p_value": permutation_p(sa, sb, n=n_perm),
                        "cliffs_delta": d, "effect_size": effect_label(d)})
            testable.append(len(rows))
        rows.append(row)
    if testable:
        adj = holm([rows[i]["p_value"] for i in testable])
        for i, p in zip(testable, adj):
            r = rows[i]
            r["p_holm"] = p
            if p < 0.05:
                leader = r["brand_a"] if r["mean_diff_a_minus_b"] > 0 else r["brand_b"]
                r["verdict"] = f"{leader} higher (Holm p<0.05)"
            else:
                r["verdict"] = "no clear difference"
    return pd.DataFrame(rows)


def aspect_table(df, scorer):
    rows = []
    for r in df.itertuples():
        for sent in split_sentences(r.text):
            for theme in tag_themes(sent):
                rows.append({"review_id": r.review_id, "brand": r.brand, "theme": theme, "sentence": sent,
                             "reviewer_type": getattr(r, "reviewer_type", "unknown")})
    asp = pd.DataFrame(rows, columns=["review_id", "brand", "theme", "sentence", "reviewer_type"])
    if len(asp):
        asp["score"] = scorer(asp["sentence"])
    return asp


def theme_by_brand(asp):
    out = []
    for (brand, theme), g in asp.groupby(["brand", "theme"]):
        lo, hi = bootstrap_ci(g["score"])
        out.append({"brand": brand, "theme": theme, "mentions": len(g), "mean_sentiment": g["score"].mean(),
                    "ci_low": lo, "ci_high": hi, "pct_negative": (g["score"] <= NEG).mean() * 100,
                    "low_n": len(g) < MIN_THEME_MENTIONS})
    return pd.DataFrame(out).sort_values(["theme", "mean_sentiment"], ascending=[True, False]).reset_index(drop=True)


def shared_weak_themes(tb):
    """Themes whose mean sentiment is below zero for every brand with enough mentions."""
    ok = tb[~tb["low_n"]]
    return sorted(t for t, g in ok.groupby("theme") if len(g) > 1 and (g["mean_sentiment"] < 0).all())


def validate(df):
    report = {}
    if "rating" in df and df["rating"].notna().sum() > 3:
        d = df.dropna(subset=["rating"])
        report["spearman_vs_rating_overall"] = round(spearman(d["score"], d["rating"]), 3)
    if "label" in df and df["label"].notna().sum() > 3:
        d = df.dropna(subset=["label"])
        report["hand_labeled_reviews"] = len(d)
        report["accuracy_vs_hand_labels"] = round((d["pred"] == d["label"]).mean(), 3)
        f1s = []
        for c in ["positive", "neutral", "negative"]:
            tp = ((d["pred"] == c) & (d["label"] == c)).sum()
            p = tp / max((d["pred"] == c).sum(), 1)
            r = tp / max((d["label"] == c).sum(), 1)
            f1s.append(0 if p + r == 0 else 2 * p * r / (p + r))
        report["macro_f1_vs_hand_labels"] = round(float(np.mean(f1s)), 3)
        report["confusion"] = pd.crosstab(d["label"], d["pred"])
    return report


# -------------------------------------------------------------------- charts
def charts(bsum, tb, df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    brands = list(bsum["brand"])
    colors = dict(zip(brands, ["#17324d", "#2f6f8f", "#7fd1c7", "#c98b3a", "#8f5f8f"]))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    err = np.nan_to_num([bsum["mean_sentiment"] - bsum["ci_low"], bsum["ci_high"] - bsum["mean_sentiment"]])
    labels = [f"{b}\n(n={n})" for b, n in zip(bsum["brand"], bsum["reviews"])]
    ax.bar(labels, bsum["mean_sentiment"], yerr=err, capsize=4, color=[colors[b] for b in bsum["brand"]])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean sentiment (-1 to 1)")
    ax.set_title("Overall sentiment by brand (95% bootstrap interval)")
    fig.tight_layout(); fig.savefig(out / "sentiment_by_brand.png", dpi=160); plt.close(fig)

    if len(tb):
        themes = list(THEMES)
        themes = [t for t in themes if t in set(tb["theme"])]
        fig, ax = plt.subplots(figsize=(9, 5))
        h = 0.8 / max(len(brands), 1)
        for i, b in enumerate(brands):
            g = tb[tb["brand"] == b].set_index("theme").reindex(themes)
            y = np.arange(len(themes)) + i * h
            err = np.nan_to_num([g["mean_sentiment"] - g["ci_low"], g["ci_high"] - g["mean_sentiment"]])
            ax.barh(y, g["mean_sentiment"].fillna(0), height=h, xerr=err, color=colors[b], label=b,
                    alpha=0.9, error_kw={"lw": 0.8})
        ax.set_yticks(np.arange(len(themes)) + h * (len(brands) - 1) / 2)
        ax.set_yticklabels(themes)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Mean sentiment (-1 to 1), 95% bootstrap interval")
        ax.set_title("Sentiment by theme and brand")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "sentiment_by_theme.png", dpi=160); plt.close(fig)

    if "date" in df and df["date"].notna().any():
        d = df.dropna(subset=["date"]).assign(month=lambda x: x["date"].dt.to_period("M").astype(str))
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for b in brands:
            m = d[d["brand"] == b].groupby("month")["score"].mean()
            if len(m):
                ax.plot(m.index, m.values, marker="o", label=b, color=colors[b])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Mean sentiment"); ax.set_title("Sentiment over time")
        ax.legend(fontsize=8)
        plt.xticks(rotation=45); fig.tight_layout(); fig.savefig(out / "sentiment_over_time.png", dpi=160); plt.close(fig)


# ------------------------------------------------------------------ pipeline
def run_analysis(raw, model="vader", min_n=MIN_N_DEFAULT, permutations=5000):
    """Run the full analysis and return every table as a dict of DataFrames."""
    df = clean(raw)
    scorer = make_scorer(model)
    df["score"] = scorer(df["text"])
    df["pred"] = df["score"].map(label_from_score)
    asp = aspect_table(df, scorer)
    empty_tb = pd.DataFrame(columns=["brand", "theme", "mentions", "mean_sentiment", "ci_low",
                                     "ci_high", "pct_negative", "low_n"])
    return {
        "df": df,
        "asp": asp,
        "counts": source_counts(df),
        "bsum": brand_summary(df, min_n),
        "tb": theme_by_brand(asp) if len(asp) else empty_tb,
        "h2h": head_to_head(df, min_n, permutations) if df["brand"].nunique() > 1 else pd.DataFrame(),
        "validation": validate(df),
    }


def save_outputs(res, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    res["df"].to_csv(out / "reviews_scored.csv", index=False)
    res["asp"].to_csv(out / "aspect_sentences.csv", index=False)
    res["counts"].to_csv(out / "source_counts.csv", index=False)
    res["bsum"].to_csv(out / "brand_summary.csv", index=False)
    res["tb"].to_csv(out / "theme_by_brand.csv", index=False)
    res["h2h"].to_csv(out / "head_to_head.csv", index=False)
    try:
        charts(res["bsum"], res["tb"], res["df"], out)
    except ImportError:
        print("matplotlib not installed, skipping PNG charts")


def print_report(res, model, min_n):
    df, asp, counts, bsum, tb, h2h = (res[k] for k in ("df", "asp", "counts", "bsum", "tb", "h2h"))
    pd.set_option("display.width", 200)
    print(f"{len(df)} reviews after cleaning, {len(asp)} theme mentions, model={model}\n")
    print("Reviews by brand and source:\n" + counts.to_string(index=False), "\n")
    print("Brand summary:\n" + bsum.round(2).to_string(index=False), "\n")
    small = bsum[bsum["reviews"] < min_n]["brand"].tolist()
    if small:
        print(f"WARNING: below the {min_n}-review minimum, no verdicts for: {', '.join(small)}\n")
    if len(h2h):
        print("Head-to-head:\n" + h2h.round(3).to_string(index=False), "\n")
    if len(tb):
        print("Theme by brand (mean sentiment):")
        print(tb.pivot(index="theme", columns="brand", values="mean_sentiment").round(2).to_string(), "\n")
        weak = shared_weak_themes(tb)
        print("Themes negative for every brand:", ", ".join(weak) if weak else "none", "\n")
    for k, v in res["validation"].items():
        print(f"{k}:\n{v}\n" if k == "confusion" else f"{k}: {v}")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--model", choices=["vader", "transformer"], default="vader")
    ap.add_argument("--min-n", type=int, default=MIN_N_DEFAULT,
                    help="minimum reviews per brand before a verdict is reported")
    ap.add_argument("--permutations", type=int, default=5000)
    a = ap.parse_args()

    res = run_analysis(pd.read_csv(a.input), a.model, a.min_n, a.permutations)
    save_outputs(res, a.out)
    print_report(res, a.model, a.min_n)
    print("\nTip: run compare.py to build the interactive dashboard.")


if __name__ == "__main__":
    main()