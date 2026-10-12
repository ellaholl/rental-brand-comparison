# Review sentiment analysis

Two tools that share one pipeline (`analyze.py`):

- `analyze.py` scores one set of reviews and reports sentiment by theme.
- `compare.py` compares several brands side by side, such as **Nuuly, BNTO and Rent the Runway**, and writes an
  interactive `dashboard.html`.

`data/sample_*.csv` is **synthetic example data written for testing** (brands are called Brand A, B and C).
It is not real reviews. The dashboard shows a warning banner whenever it sees synthetic data.

## Run
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python compare.py --input data/sample_compare.csv --out outputs   # open outputs/dashboard.html
pytest
```

## Compare Nuuly, BNTO and Rent the Runway
1. Copy `data/reviews_template.csv` to `data/rental_reviews.csv`.
2. Add one row per review with the same columns as the sample: `brand, source, date, rating, reviewer_type, label, text`. Only `brand` and `text` are required. `label` (your own positive, neutral or negative label) lets the tool check itself. Use the brand names `Nuuly`, `BNTO`,
   `Rent the Runway`.
3. Run `python compare.py --input data/rental_reviews.csv --out outputs`.

Aim for at least 30 reviews per brand. BNTO is a newer, smaller service with far fewer public reviews than the other
two, so expect wide intervals for it. The tool refuses to give a verdict for any brand under `--min-n` reviews
(default 30) and says so on the dashboard. Do not pad BNTO with reviews from other sources just to reach the number.
Collect the same kinds of sources for each brand, or note the difference.

## Metrics
Per brand: review count, mean sentiment with a 95% bootstrap interval, percent positive / neutral / negative,
net sentiment (positive minus negative), average star rating, and agreement between sentiment and stars.

Per theme (fit and sizing, condition and quality, delivery and returns, selection and style, pricing and value,
subscription, customer service, app and website): mean sentiment of the sentences that mention it, with the number
of mentions. Cells with fewer than 5 mentions are greyed out.

Per pair of brands: difference in mean sentiment with a bootstrap interval, a permutation-test p-value corrected
with the Holm method, and the Cliff's delta effect size.

## Collecting data responsibly
- Use reviews that are public and that you may use. Check each site's terms first. Many forbid scraping.
- Prefer manual collection, official exports or documented APIs, and cite each source.
- Keep only text, date, rating, source and brand. Do not store names or other personal details.
- Reviews are copyrighted by their authors. Publish aggregate results and short paraphrases, not full dumps.

## Known limits
Keyword theme tagging misses indirect wording. VADER can misread sarcasm. Reviews are self-selected and skew
negative. Different sites attract different customers, so a gap between brands may reflect where reviews were
collected. Validate against hand labels by adding a `label` column and running `analyze.py`.
