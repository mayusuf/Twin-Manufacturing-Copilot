# EDA slide takeaways

Input: `data/processed/model_input.csv` (10,000 observations).
Source: UCI AI4I 2020 Predictive Maintenance Dataset, DOI `10.24432/C5HS5C`.
The dataset is synthetic. These are descriptive associations in the full dataset.

## 1. Machine failure class distribution

Failures: 339 (3.39%).
No failure: 9,661 (96.61%).

Slide takeaway: Strong class imbalance. A model that always predicts no failure
would have 96.61% accuracy and zero failure recall.
Use a stratified split and compare failure recall, precision, F1, and PR-AUC.

## 2. Failure rate by product quality

L = low, M = medium, H = high product quality.

- Quality L: 3.92% (235 failures / 6,000 observations).
- Quality M: 2.77% (83 failures / 2,997 observations).
- Quality H: 2.09% (21 failures / 1,003 observations).

Slide takeaway: Quality L has the highest observed failure rate.
Keep product quality as a categorical predictor. The feature builder uses a fixed,
predefined L/M/H one-hot vocabulary; it learns no encoding statistics from data.

## 3. Numeric features by failure status

Medians (failure vs. no failure):

- Air temperature (K): 301.6 vs. 300.0.
- Process temperature (K): 310.4 vs. 310.0.
- Rotational speed (rpm): 1365.0 vs. 1507.0.
- Torque (Nm): 53.7 vs. 39.9.
- Tool wear (min): 165.0 vs. 107.0.

Slide takeaway: Failure observations have higher median torque and tool wear,
and lower median rotational speed. The distributions still overlap.
Extreme values are retained because they may describe failure conditions.

## 4. Torque vs. rotational speed

Pearson correlation across all observations: -0.875.

Slide takeaway: Torque and rotational speed have a strong inverse association.
Failure points also occur across multiple operating regions. Investigate their
interaction during feature engineering; this chart does not establish causation.

## Output and modeling notes

- Figures are 3200 x 1800 PNGs (16:9) with matching PDF exports.
- Counts, rates, and feature statistics are saved in `reports/tables`.
- EDA used the full dataset. Freeze modeling choices before evaluating the test set.
- The fixed L/M/H encoding is shared by training and twin inference.
- Fit any learned transformations or resampling using training data only.
