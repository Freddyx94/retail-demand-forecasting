# Findings

**Scope:** Walmart store CA_1, FOODS category (1,437 products), 29 Jan 2011 to 22 May 2016.
**Source:** every number below can be reproduced. Analysis findings come from [`analysis/01_eda.ipynb`](analysis/01_eda.ipynb) (section shown in brackets; run `dbt build` first). Model results come from the scripts named in each section.

## Key findings at a glance

| # | Finding | What it means for the forecast model |
|---|---|---|
| 1 | Growth comes from a wider range, not from busier products; clear summer peak | Features that follow each product's *recent* level; month; days since launch |
| 2 | Weekends sell about 1.5× a midweek day, almost every week | Weekday feature; lags in whole weeks (28, 35, 42 days) |
| 3 | SNAP (food-stamp) days +13%; individual holidays move sales by up to ±25%, and the day *before* a holiday by up to +48% | SNAP and day-of-month flags; event *name*; "days before event" features |
| 4 | Customers react to price (elasticity ≈ −2.3), but prices rarely change | Price features based on *past* prices only |
| 5 | Most products sell nothing on half their days; 10% of products make half the volume | Model built for counts with many zeros; error reported by product group |
| 6 | 15% of the data looks like out-of-stock periods recorded as zero sales | Remove these days from training targets; evaluate with and without them |

---

## 1. Growth comes from more products, not busier products [section 1]

- Total FOODS sales rose over the five years, but the number of products on sale more than doubled (628 → 1,437).
- Following only the 628 products that were on sale from the start, average sales per product **fell about 24%** between 2012 and 2015. None of them stopped selling completely.
- After removing the trend, there is a clear yearly pattern: **June to September 6–12% above normal** (August highest), **January to March 5–7% below**.
- The store is **closed on 25 December** every year (sales = 0).

**Decision:** features that track each product's recent sales level (rather than one long-term average), plus month, days since launch, and a "store closed" flag for Christmas.

## 2. A strong, regular weekly cycle [section 2]

- **Saturday +25% and Sunday +29%** compared with the average day of the same week; **Tuesday to Thursday 16–18% below**. A weekend day sells **1.53×** a midweek day.
- Saturday or Sunday was the best day in **96% of 272 complete weeks** (Friedman test, p < 10⁻²⁰⁰).
- Sales are most similar to sales exactly 1, 4 or 5 weeks earlier (autocorrelation 0.74 at 7 days, 0.81 at 28 days, but only 0.44 at 27 days).

**Decision:** weekday as a feature; past-sales features in whole weeks starting at 28 days (28, 35, 42), so the model never uses information it would not have when making a 28-day forecast.

## 3. SNAP days and specific holidays move sales [section 3]

- On **SNAP days** (the 10 days a month when California pays food-stamp benefits), sales per product are **12.9% higher** (95% CI 11.7–14.2%), after allowing for weekday and month. The effect is strongest in FOODS_2 (+24%), smaller in FOODS_3 (+12%) and absent in FOODS_1.
- *Caveat:* SNAP days are always the 1st to 10th of the month in California, so SNAP cannot be fully separated from a general beginning-of-month effect.
- **Holidays matter by name, not by type.** Lower sales on New Year's Day (−24%), Thanksgiving (−19%), Mother's Day (−18%), Easter and Halloween (−15%). Higher sales on Monday public holidays such as Labor Day (+24%) and Memorial Day (+13%).
- **The day before often matters more:** the day before Thanksgiving is **+48%**, and two days before Christmas is +29%.
- *Caveat:* each holiday occurs only 4–6 times, and with 30 holidays tested 1–2 "significant" results are expected by chance. Only the large, consistent effects are listed.

**Decision:** SNAP and day-of-month features; use the holiday name instead of its type; add "days until the next event". Holidays are known in advance, so these features do not leak future information.

## 4. Customers react to price, but prices rarely change [section 4]

- Within the same product, a **10% lower price goes with about 25% more units** (elasticity −2.3, 95% CI −2.7 to −1.9). This is similar across the three departments.
- Price changes are rare: only **1.3% of product-weeks** have one, and **312 products (22%) never changed price**. Most changes are permanent: only 13% of price drops returned to the old price within four weeks.
- *Caveat:* prices are not changed at random (a price cut may come with a display or advert), so −2.3 is an association, not a proven cause.

**Decision:** price, price compared with the product's *recent past* average, and week-on-week price change. Using the average over the whole history would leak future prices into the model.

## 5. Most products sell irregularly; a few carry the volume [section 5]

- The typical product sells **nothing on 49% of its days**. **85%** of products have intermittent or "lumpy" demand (Syntetos-Boylan classification).
- **141 products (10%) make up half of all units sold.** Fast movers (≥ 5 units/day) are 9% of products but 48% of volume; slow movers (< 1/day) are 45% of products but 11% of volume.

**Decision:** LightGBM with a Tweedie objective (designed for counts with many zeros); RMSSE as the main metric so slow and fast products count equally; errors and bias reported by speed group, plus a volume-weighted error.

## 6. 15% of the data looks like stockouts, not zero demand [section 6]

- There are **3,288 long runs of zero sales** (28+ days) that are statistically very unlikely given each product's normal sales rate (probability < 1 in a million). They affect **79% of products** and cover **15.3% of all rows**.
- Example: the store's best-selling FOODS product (FOODS_3_090, about 67 units a day) sold **nothing for 227 days** in 2011.
- The data has no stock levels, so this cannot be proven. But in store and warehouse systems a product that is out of stock is recorded as zero sales, not as missing demand, which is the most likely explanation.

**Decision:** flag these days as *likely out of stock* and **exclude them from the training targets**, so the model does not learn that a best seller sometimes has zero demand for months. Report accuracy both with and without these days. The flag needs the full history, so it is used only to clean training data, never as a feature at forecast time.

---

## Model results

**Task:** forecast daily units for each of the 1,437 products, 28 days ahead.
**Main score:** RMSSE, the forecast error relative to each product's normal day-to-day variation, averaged over products, so slow and fast sellers count equally (lower is better). **Bias:** total forecast vs total actual sales.

### Final test (hold-out: 25 April – 22 May 2016)
These four weeks were not used for any decision. All model choices were made on three earlier 28-day test windows (rolling-origin cross-validation). Source: `python -m forecasting.final_eval` → [`results/holdout_results.csv`](results/holdout_results.csv).

| Model | What it does | RMSSE | Weighted RMSSE¹ | Avg. miss (units/day) | Bias |
|---|---|---|---|---|---|
| Seasonal naive | Repeats the last 4 weeks | 1.021 | 1.077 | 1.84 | −6.8% |
| Moving average | Average of the last 28 days | 0.779 | 0.826 | 1.52 | −6.8% |
| ETS | One statistical model per product | 0.766 | 0.797 | 1.46 | −6.7% |
| **LightGBM** | One machine-learning model for all products | **0.754** | **0.774** | **1.44** | −6.8% |

¹ Products weighted by their recent sales value, as in the M5 competition: errors on big sellers count more.

**LightGBM is 26% better than the seasonal-naive rule**, 3.2% better than a 28-day moving average and 1.5% better than ETS. Weighted by sales value, the gain over the moving average doubles to 6.4%. The cross-validation gave the same ranking and similar margins (27% / 3% / 2%), so the result is stable.

**Every model under-forecast by about 7%.** Sales rose 7.3% in the final four weeks compared with the four before (the start of the summer peak, finding 1), and none of the models anticipated it.

## 7. Most of the gain is on fast, regular sellers
Source: `python -m forecasting.error_analysis` (three CV windows; LightGBM vs moving average).

- **Fast sellers** (≥ 5 units/day): LightGBM is **8% better**, and wins for 74% of these products. **Slow sellers** (< 1/day): only **1% better**.
- **Smooth** demand: 6% better. **Intermittent and lumpy** demand: about 2%.
- Every gap holds in all three test windows and is significant (Wilcoxon signed-rank test across products, p < 10⁻⁵).

**Meaning:** the model adds value where there is a pattern to learn. For products that sell a unit every few days, no model does much better than an average.

## 8. The model is mainly a smarter moving average
Source: `python -m forecasting.error_analysis` (feature importance by gain, mean over CV windows).

- Recent sales levels make up **77%** of the model's decisions (last 28 days 44%, last 7 days 29%, last 56 days 4%). The product itself adds 11%, and the same weekday over the last 4 weeks 5%.
- Calendar, events, price and SNAP together make up **about 4%** (the remaining 3% is days since the last sale, sales variability, horizon and product age). SNAP contributes almost nothing, even though SNAP days sell 13% more (finding 3). SNAP days are always the 1st–10th of the month, so the recent averages and day of month already carry most of that information.
- Before events and on SNAP days, LightGBM's under-forecast is much smaller than the moving average's (−1.4% vs −5.9% before events, −0.4% vs −4.5% on SNAP days). On event days themselves it under-forecasts more (−5.9% vs −2.8%).

## 9. Accuracy is limited by randomness, not by model settings
Sources: `python -m forecasting.tune`, `python -m forecasting.train`.

- Seven LightGBM variants (tree size, learning rate, Tweedie power, minimum leaf size) scored within **0.5%** of each other on cross-validation, less than the variation between test windows. The default settings were kept.
- A first version whose features looked back at least 28 days per row lost accuracy: features computed on the forecast date work better. A 28-day average computed that way scored 0.740 instead of 0.717 on cross-validation.

**Meaning:** further gains need better *information*, such as stock levels, promotions or displays, rather than more tuning.

## 10. Removing likely stockouts changes the direction of the error, not its size
Sources: `python -m forecasting.train`, `python -m forecasting.final_eval`.

- Training without the likely-stockout days (finding 6) teaches the model **demand** instead of sales. In cross-validation it removed the under-forecast on in-stock days (−4.6% → +1.5%), but scored worse on all days (RMSSE 0.750 vs 0.696), because on stockout days actual sales were 0.
- In the final test, where few days were flagged, the two versions scored almost the same (RMSSE 0.756 vs 0.754), and the stockout-cleaned version had the smaller bias (+2.0% vs −6.8%).
- **The main model keeps stockout days** (it scores best on the standard measure). For ordering stock, the stockout-cleaned version is the better choice, because ordering to match past empty shelves keeps shelves empty.

## Limitations
- **One store, one category.** Results may not transfer to other stores or to non-food products.
- **No stock data.** Stockouts are inferred from improbable zero runs, not observed.
- **Calendar effects are under-used.** The model does not anticipate seasonal shifts such as the May upturn; a stronger yearly-seasonality feature is a candidate improvement.
