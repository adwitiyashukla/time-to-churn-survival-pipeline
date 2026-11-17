# time-to-churn-survival-pipeline

Churn prediction on the KKBox music streaming data, treated as a time to event problem instead of a fixed window label.

Most churn projects pick a window, say 30 days, tag everyone inside it as churned, split the rows at random and report AUC. That throws away when someone leaves. It also mislabels anyone whose window runs past the end of the data, and a random split lets the future leak backwards into training. This pipeline builds a person period panel, fits a discrete time hazard, splits by calendar date, and scores with metrics that account for censoring.

## Data

KKBox WSDM 2018 churn competition. Two files are used, `transactions.csv` and `members_v3.csv`. The 30 GB `user_logs` file is not used.

Ingest kept 21,545,970 of 21,547,746 transactions across 2,362,947 subscribers. 1,776 rows were dropped for dates that do not parse. 1,930,371 subscribers have a members record, and the rest get a missing flag instead of being dropped.

If you do not have a Kaggle account, there is a generator that writes files with the same schema and scale:

```
python -m churn_survival.cli synth
```

## How churn is defined

A membership lapses when the expiry date passes and no renewal appears within a 30 day grace period.

Two decisions here mattered more than the modelling.

When someone cancels, the clock stops at the cancellation date, not at the expiry date. If it stopped at expiry, the cancellation transaction would sit inside the feature window and the model would just be reading back a decision that had already happened.

Censoring is per subscriber. Someone paid up past the end of the data has no known outcome, so they are censored at their own paid through date rather than at one shared cutoff. That is the shape real subscription data has.

The panel came out at 19,474,454 rows over 2,272,231 subscribers, with a hazard of 7.83 percent per 30 day period. That sits on top of the churn rate KKBox published for the competition, which was a good sign the definition was right.

Of 1,524,812 churn events, 440,961 are cancellations and 1,083,851 are passive lapses. So 71 percent of churn is people quietly not renewing rather than people clicking cancel.

## Leakage

Every feature is built only from transactions strictly before the start of the period it belongs to.

Three tests check that. One rebuilds features with the transaction history truncated at a cutoff and asserts nothing moved. One corrupts every transaction at or after the cutoff and asserts not a single feature changed. One recomputes a sample of rows by brute force in a plain loop and compares against the vectorised version.

## Models

One model is fitted per horizon step. Step 1 predicts churn in the next 30 days, step 2 predicts churn in days 30 to 60 given survival to day 30, and so on. Survival is the running product of one minus each hazard.

The first version chained a single hazard model forward and carried covariates like days to expiry unchanged into future periods. That is fiction, because a renewal resets that clock, and it showed up as the naive baseline beating the survival models at every horizon. Fitting a model per step removes the problem, since no covariate ever has to be projected.

Everything is trained on periods up to 2016-07-30 and evaluated at a landmark of 2016-10-31, on 806,979 subscribers still at risk with 142,833 events observed after that date.

The `naive_fixed_window` row is the comparison that matters. Same features, same split, same model family. The only difference is that it labels a subscriber as not churned when the outcome is not yet observable, which is what you get if you build a fixed window label with a data cutoff.

| model | Uno C | integrated Brier | calibration error | AUC 30d | AUC 60d | AUC 90d |
| --- | --- | --- | --- | --- | --- | --- |
| **boosted hazard** | 0.779 | **0.068** | 0.059 | 0.859 | 0.800 | 0.785 |
| **logistic hazard** | 0.770 | 0.074 | **0.029** | 0.872 | 0.797 | 0.776 |
| **naive fixed window** | 0.772 | 0.083 | 0.079 | **0.916** | **0.823** | 0.777 |
| **Cox proportional hazards** | 0.775 | 0.102 | 0.073 | 0.842 | 0.789 | **0.802** |
| **Kaplan-Meier** | 0.500 | 0.120 | 0.072 | 0.500 | 0.500 | 0.500 |

The boosted hazard beats the naive baseline by 18 percent on integrated Brier score. The logistic hazard is 2.7 times better calibrated. The naive model still wins raw AUC at 30 days, which is what you would expect from something trained directly on a 30 day question, but its probabilities are the least trustworthy of the four.

Cox ranks well at 90 days and has the worst Brier score of any real model here. The Schoenfeld residual test gives p values down to 1e-103, so proportional hazards does not hold on this data, and the calibration reflects that.

Metrics are written from scratch rather than pulled from a package. Kaplan-Meier and the reverse Kaplan-Meier censoring curve, IPCW Brier and integrated Brier, Uno's C on a Fenwick tree, time dependent AUC, and calibration measured against a within bin Kaplan-Meier estimate rather than a raw event rate. They were checked against a simulation where the true survival function is known.

## Competing risks

Cancellations and passive lapses are modelled as separate cause specific hazards, combined into cumulative incidence.

| cause | predicted at 90 days | observed at 90 days |
| --- | --- | --- |
| **voluntary cancel** | 0.034 | 0.038 |
| **passive lapse** | 0.118 | 0.139 |
| **total** | 0.151 | 0.177 |

The split between the two causes comes out about right. Total churn is under predicted by roughly 15 percent relative, which is where this could be improved next.

## Who to contact

Survival curves give expected remaining lifetime, so every subscriber gets an expected residual value as well as a churn probability. Mean expected remaining lifetime at the landmark is 5.04 periods and mean residual margin is 404 NTD.

Both policies contact the same 80,698 people at the same cost.

| policy | churners caught | precision | value saved | net gain |
| --- | --- | --- | --- | --- |
| **value at risk** | 39,266 | 0.487 | 2,823,181 | **2,702,134** |
| **churn probability** | 65,922 | 0.817 | 1,520,332 | 1,399,285 |
| **random** | 14,109 | 0.175 | 848,511 | 727,464 |

Ranking by churn probability catches 68 percent more churners and makes 46 percent less money. The subscribers who are easiest to spot as leaving are the cheap ones. Ranking by expected value at risk gives up precision and nearly doubles net gain.

## Running it

Every tunable lives in `config.toml` and is validated on load.

```
python -m churn_survival.cli ingest
python -m churn_survival.cli panel
python -m churn_survival.cli features
python -m churn_survival.cli evaluate
python -m churn_survival.cli decide
```

Or `all` to run them in order. Ingest is chunked and the panel is built with numpy on sorted arrays, so the whole thing runs inside 4 GB on 21 million transactions. On an 8 GB laptop it takes about 15 minutes end to end, most of that in evaluate.

Tests:

```
python -m pytest tests -q
```

## Layout

```
churn_survival/
  config.py     load and validate config.toml
  synth.py      generate a stand in dataset with the real schema
  ingest.py     chunked csv to parquet
  panel.py      spells and the person period panel
  features.py   as of time features
  splits.py     calendar split and landmark cohorts
  models.py     the model ladder
  metrics.py    survival metrics
  evaluate.py   scoring and the naive comparison
  decide.py     customer value and targeting
  cli.py        stage entry points
tests/
```
