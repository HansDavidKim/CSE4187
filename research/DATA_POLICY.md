# Data-use policy for the KSC submission

This is an engineering record, not legal advice.

## Approved paper sources

1. **Criteo Uplift v2.1** is published by the verified Criteo organization under
   **CC BY-NC-SA 4.0**. It contains 13.9M randomized-control-trial rows with 12 anonymized
   pre-treatment features, treatment, visit, conversion and exposure. Cite the dataset and
   Diemert et al., *A Large Scale Benchmark for Individual Treatment Effect Prediction and
   Uplift Modeling*. Do not commit or redistribute the raw file. Mark any distributed adapted
   dataset with the same license and attribution; keep use non-commercial.
2. **UCI Online Retail** is published under **CC BY 4.0**. Cite the UCI DOI/source and the
   associated publication. It supplies customer/time purchase trajectories, not randomized
   interventions.

## Starbucks status

The Kaggle API reports `licenseName: Other (specified in description)`. The description says the
mock data was made available by Starbucks to Udacity scholars for projects, but it contains no
actual grant covering research publication, adaptation or redistribution. Therefore:

- local exploratory analysis is retained;
- no raw rows are committed or attached to a paper artifact;
- no headline KSC result depends on this copy;
- promotion to a paper dataset requires written permission or a source with explicit terms.

### Official Udacity promotion exercise

The separate `DSND_Term2/portfolio_exercises/Experiments/data` source contains `training.csv` and
`Test.csv` with Promotion, purchase and V1–V7. The repository root LICENSE is MIT, while its README
states CC BY-NC-ND 4.0 for the educational work and links to Udacity terms. Neither statement has
been verified as a dataset-specific redistribution grant from Starbucks. The project therefore
downloads pinned files locally, records checksums/source commit, publishes no original or modified
rows, and uses the data only for non-commercial academic model calibration. The checked official
notebook describes a Starbucks take-home promotion exercise but does not explicitly document
random assignment; do not call it an RCT without a separate authoritative source.

## Required wording

Do not call observational offer logs interventions. Use `logged action` unless random assignment
or another identification design is documented. Report causal estimates only for the randomized
Criteo arm or synthetic known-truth experiments; label Online Retail/Starbucks results as
behavioral calibration or associational discovery.

