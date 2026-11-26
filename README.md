Domain: Support

# Swiggy Support Decision Intelligence (capstone)

Synthetic support-ticket data (800 records, 7 predictors, an escalation outcome and a resolution-time value). Everything is plain Python and NumPy; every graded number is computed by code in this repository and written to `outputs/`. The data is synthetic and says nothing about Swiggy's real tickets.

## How to run

From a clean clone, in this order:

```
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

`generate_data.py` prints `800 10 170 7cb48ef89914049d76954530b6482e04`. `common.py` and `models.py` are shared helpers; importing them does not run a Part. Each `partN.py` regenerates its `outputs/partN_*` files.

## Part 1 findings

**R1.1** H0 (mean resolution time = 8.0 minutes) is rejected: `z_stat` = 2.8930, which is beyond the critical value 1.96 (the 95% interval 8.3026 to 9.5738 also excludes 8.0). A Type I error here would mean concluding that the average resolution time is no longer 8 minutes (and re-planning staffing) when it truly still is 8. The gap `mean_value` − 8.0 is 0.9382 minutes, about 56 seconds or 11.7% above the plan. It is statistically clear, but per ticket it is small; it matters for staffing only because it is multiplied across a very large number of tickets.

**R1.2** `n_required` = 1293 tickets for a half-minute margin of error at 95% confidence, against 800 rows available. The current margin is about 0.64 minutes, so the team would need roughly 493 more tickets before it can claim half-minute precision.

**R1.3** `gd_slope_gap` = 0.0000 (below 0.001). Least squares is a convex quadratic cost with one minimum, and gradient descent with a suitable step size walks to that same minimum (the cost stops changing by about update 100).

**R1.4** `resid_sd_ratio_raw` = 3.7525 and `resid_sd_ratio_log` = 1.0396. The raw model shows a clear funnel (residual spread in the upper half is almost four times the lower half), and the log-log model removes it (ratio close to 1).

**R1.5** Predictors with |coefficient| above 0.05 (mechanical rule, not an importance ranking): ln(X4) (1.1576), X2 (0.0940), X3 (0.1067) and X6 (0.1518). `mr_adj_r2` = 0.7403 against `loglog_adj_r2` = 0.6561, so the extra predictors earn their place (adjusted R² rises by about 0.084 even after the penalty for more variables).

## Part 2 findings

**R2.1** After 200 updates the losses are, lowest to highest: η = 1.0 (0.3703), η = 0.1 (0.3776), η = 0.01 (0.5091). A rate that is too small barely moves in 200 steps, while the larger rate converged fastest here (a large rate can also overshoot and diverge on other problems, which this data did not trigger).

**R2.2** The three largest absolute weights are X3 (0.9013), X1 (−0.5554) and X2 (0.5505). The largest odds ratio is X3 = 2.4629: each one-SD rise in the share of the resolution-time target already used multiplies the odds that a ticket escalates by about 2.5, other predictors held fixed.

**R2.3** `blind_accuracy` = 0.7875 against 0.8042 accuracy at τ = 0.5: predicting "no escalation" for everything almost matches the model, so accuracy hides the fact that the model finds only 0.4706 of escalations. Lowering τ from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 and lowered precision from 0.5455 to 0.4634 (F1 rose from 0.5053 to 0.5714, accuracy fell to 0.7625).

**R2.4** At τ = 0.3 the logistic model has `f1` 0.5714 and recall 0.7451; the KNN model (K = 5) has `f1` 0.3696 and recall 0.3333. Logistic regression is clearly better on both.

**R2.5** I would deploy the logistic model at τ = 0.15 (`cost_tau`). On the test rows it costs INR 6050 (`cost_test_at_tau`), against INR 11800 at τ = 0.5 and INR 20400 if nobody is flagged. The threshold was chosen on training rows so the test rows stay an untouched estimate of future cost; picking it on the test rows would make the reported cost optimistic.

## Part 3 findings

**R3.1** WCSS drops: 2→3 = 659.8282, 3→4 = 108.0739, 4→5 = 171.0575, 5→6 = 45.5466. The big drop is from 2 to 3 clusters, and `elbow_k` = 3 (the 3→4 drop is below half of the 2→3 drop). The drops do not shrink steadily (4→5 is larger than 3→4) because the pinned starting rows give one reproducible solution, not necessarily the global optimum.

**R3.2** (rows of `part3_segments.csv`)
- **Segment A, "Quick routine tickets"** (322 tickets, mean SLA used 0.2250, 1.95 earlier tickets, mean value 3.60 minutes, `outcome_rate` 0.0186). Action: keep them in a self-service or low-touch queue with no supervisor review.
- **Segment B, "Repeat-contact, high-value, slow tickets"** (255 tickets, 6.07 earlier tickets, mean X4 28.7 thousand, mean value 17.12 minutes, `outcome_rate` 0.2353). Action: route to experienced agents and plan extra handling time in staffing.
- **Segment C, "Late-arriving, near-deadline tickets"** (223 tickets, mean SLA used 0.7118, `outcome_rate` 0.4664). Action: triage first and trigger an early supervisor review.

Segment C has the highest `outcome_rate`, 0.4664.

**R3.3** Single linkage cuts into 4 clusters: {E1}, {E2, E3, E6}, {E4}, {E5}. Complete linkage cuts into 2 clusters: {E1, E4, E5} and {E2, E3, E6}. Single linkage joins on the closest pair, so chains form with small steps (the last merge is only 3.4) and the biggest gap shows up early; complete linkage joins on the farthest pair, so the two groups stay far apart (final merge 7.9) and the biggest gap comes at the end.

**R3.4** The mean vector is all zeros after standardising. PC1 is driven by X1 (+0.5317), X3 (−0.5187), X5 (+0.4418) and X2 (−0.4100), with X4 (−0.2916) smaller: it separates long-tenure, satisfied customers from those with many earlier tickets and ticket deadlines already largely used. Three components are needed for 80% of the variance (`cumulative_ratio_2` 0.7593, `cumulative_ratio_3` 0.8616). `mean_pc1_segment_A/B/C` = 1.7773 / −0.7831 / −1.6709: PC1 orders the segments in the same order as their escalation rates (lowest in A, highest in C).

## Part 4 findings

**R4.1** The best split is S4 (tenure ≤ 24 months): it has the highest information gain (0.1252) and Gini gain (0.0606) because it separates the records into a side with a 45.7% escalation rate (86 of 188) and a side with 8.9% (33 of 372). A tree grown until every leaf is pure would memorise noise and individual random outcomes, so training error would be zero but new tickets would be predicted poorly (overfitting). A maximum depth, a minimum number of records per leaf or a minimum gain per split stops a tree from growing too deep.

**R4.2** I would ship the logistic model (`f1` 0.5714, `recall` 0.7451). The stump has `f1` 0.5532 and `recall` 0.7647, hard vote 0.5344 and 0.6863, soft vote 0.5507 and 0.7451, KNN 0.3696 and 0.3333. Combining models did not help: neither vote beat the logistic model on F1.

**R4.3** After round 1 the two misclassified points (3 and 7) carry a total weight of 0.2500 + 0.2500 = 0.5000. Round 2 trusts h2 more because its weighted error (0.1667) is lower than h1's (0.2500), so α2 = 0.8047 is larger than α1 = 0.5493.

**R4.4** Model A option 2 (convolutional network):

| Layer | Output shape (height × width × depth) | Parameters |
|---|---|---|
| Input | 28 × 28 × 1 | 0 |
| Conv 5×5, 6 filters, pad 2 (+ReLU) | 28 × 28 × 6 | 156 |
| Max-pool 2×2, stride 2 | 14 × 14 × 6 | 0 |
| Conv 5×5, 16 filters, pad 0 (+ReLU) | 10 × 10 × 16 | 2416 |
| Max-pool 2×2, stride 2 | 5 × 5 × 16 | 0 |
| Flatten | 400 × 1 × 1 | 0 |
| Dense 32 (+ReLU) | 32 × 1 × 1 | 12832 |
| Dense 10 (output) | 10 × 1 × 1 | 330 |

`cnn_total` = 15734 against `mlp_total` = 13002 for option 1.

Model B shapes: W_xh is 128 × 64, W_hh is 128 × 128 and W_hy is 16 × 128 (plus b_h of 128 and b_y of 16), giving `rnn_total` = 26768. It does not grow with longer sequences because the same weights are reused at every time step.

For 10 classes I would use ReLU in the hidden layers and softmax at the output with categorical cross-entropy loss.

In a very deep plain network the gradient is a product of many per-layer factors, and when those are below 1 it shrinks layer by layer, so the early layers barely learn. A skip connection adds a layer's input to its output, giving the gradient a direct additive path back to the early layers.

## Model-choice memo

To: Head of Customer-Support Operations

I recommend the logistic-regression model for production. It had the best F1 of the five structured models on the 240 test tickets (0.5714 at τ = 0.3), against 0.5532 for the one-question stump, 0.5344 for hard voting, 0.5507 for soft voting and 0.3696 for nearest neighbours. At τ = 0.3 it catches 38 of the 51 test escalations (recall 0.7451). Combining models added no benefit, so there is no reason to run three models.

Threshold: alert at 0.15. It was chosen on training tickets by minimising cost, where a needless supervisor review costs INR 50 and a missed escalation costs INR 400. On the test tickets this costs INR 6050, against INR 11800 at the default threshold of 0.5 and INR 20400 if nobody is flagged. That is a saving of about INR 14350 per 240 tickets compared with flagging nobody. Expect many more reviews at 0.15 than at 0.5, so the review team needs capacity for them.

Risk: the evidence rests on one split with only 51 escalations in the test set, on synthetic data, and on assumed rupee costs; the true savings could differ noticeably.

Model B (ticket-text sorting) is the one to build next, but neither Model A nor Model B is worth building before the structured model is in production.

## Limitations

1. The data is synthetic and generated from known formulas, so real tickets may behave differently and results may not transfer.
2. All model comparisons use a single 70/30 split with only 51 escalations in the test set, so metrics and costs are noisy and the best K and threshold are valid for this split only.
3. The costs of INR 50 and INR 400 are assumptions; a different cost ratio would move the best threshold.
4. K-means uses pinned starting rows, so the segments are reproducible but not guaranteed to be the global optimum, and the hierarchical-clustering and AdaBoost tasks use small supplied tables rather than the ticket data.
5. No confidence intervals are reported for the classification metrics, and the networks in Model A and Model B are only sized, not trained.
