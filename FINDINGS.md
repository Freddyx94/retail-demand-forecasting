# Findings

**Scope:** Walmart store CA_1, FOODS category (1,437 products), 29 Jan 2011 to 22 May 2016.
**Source:** every number below comes from [`analysis/01_eda.ipynb`](analysis/01_eda.ipynb) (section shown in brackets) and can be reproduced with `dbt build` followed by running the notebook.

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

*To be added in Stage 3.*
