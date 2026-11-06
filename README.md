# Bank campaign modeling

The notebook explores classification and profit curves for a personal equity
plan campaign. Its input datasets are external ARFF files, not included Excel
workbooks. `campaign_baseline.py` provides a separate, reproducible evaluation
on the training and new-customer data referenced by the notebook.

## Run

Obtain `bank-training.arff` and `bank-NewCustomers.arff` from the URLs in the
notebook, subject to the source's terms, and store them locally. Then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python campaign_baseline.py bank-training.arff bank-NewCustomers.arff
python -m unittest discover -s tests -v
```

The script fits logistic regression on the training data, with numeric
imputation/scaling and categorical imputation/one-hot encoding fitted only
on that data. It evaluates ROC AUC and observed campaign profit on the labeled
new-customer file. The default payoff assumes a $100 gain for a positive target
and a $70 loss for a negative target. Under these assumptions, the model
targets probabilities strictly above `70 / (100 + 70)`; override the amounts
with `--benefit` and `--loss`. The report also shows contact-all and contact-none
profits for comparison. These are sample outcomes, not guaranteed profit.

## Limitations

The threshold assumes calibrated probabilities and the stated payoff model;
neither calibration nor incremental treatment effect is established. Outcomes
may include people who would sign up without a campaign. The data are small and
their provenance, independence, timing, representativeness and permission to use
them should be verified before drawing conclusions. This analysis uses sensitive
and demographic predictors and is not appropriate for deployment without
privacy, fairness, regulatory and independent validation review.

The notebook is a separate exploratory analysis: it chooses models and
thresholds from training cross-validation and reports test outcomes. Its
profit-maximizing threshold and apparent test-set optimum are retrospective,
and lift is not the same metric as classification accuracy. Notebook execution
requires additional dependencies and access to its external sources.
