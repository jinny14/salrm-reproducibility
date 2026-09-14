# SALRM Main Experiment Results

Generation time(UTC): 2026-08-30T16:18:48.525085+00:00

## Latent Reliability and Rankings

|   rank | model                     |   latent_reliability_baseline |   latent_reliability_safety |   condition_effect |   condition_sensitivity |   sensitivity_adjusted_reliability |   ci_low_adjusted_reliability |   ci_high_adjusted_reliability |   bootstrap_mean |   ci_low_bootstrap |   ci_high_bootstrap |   rank1_probability | vb_converged   |     n |   unique_items |
|-------:|:--------------------------|------------------------------:|----------------------------:|-------------------:|------------------------:|-----------------------------------:|------------------------------:|-------------------------------:|-----------------:|-------------------:|--------------------:|--------------------:|:---------------|------:|---------------:|
|      1 | gpt-4.1-mini              |                      0.904568 |                    0.91626  |          0.0116918 |               0.0116918 |                           0.894054 |                      0.885346 |                       0.902402 |         0.712673 |           0.698454 |            0.726144 |                   1 | True           | 15476 |           7738 |
|      2 | qwen2.5:7b-instruct       |                      0.819356 |                    0.838463 |          0.0191066 |               0.0191066 |                           0.803849 |                      0.790945 |                       0.816619 |         0.62457  |           0.606194 |            0.642863 |                   0 | True           | 15476 |           7738 |
|      3 | claude-haiku-4-5-20251001 |                      0.772936 |                    0.813784 |          0.0408481 |               0.0408481 |                           0.741999 |                      0.734668 |                       0.74937  |         0.642974 |           0.629144 |            0.65586  |                   0 | True           | 15476 |           7738 |
|      4 | llama3.1:8b               |                      0.647005 |                    0.688447 |          0.0414426 |               0.0414426 |                           0.620739 |                      0.604404 |                       0.637188 |         0.558123 |           0.540039 |            0.575643 |                   0 | True           | 15476 |           7738 |

## Uncertainty in Latent Reliability

| model                     |   latent_reliability_baseline |   ci_low_baseline |   ci_high_baseline |   latent_reliability_safety |   ci_low_safety |   ci_high_safety |   condition_effect |   ci_low_condition_effect |   ci_high_condition_effect |   posterior_probability_condition_effect_positive |   condition_sensitivity |   posterior_mean_condition_sensitivity |   posterior_sd_condition_sensitivity |   condition_sensitivity_upper_95 |   sensitivity_adjusted_reliability |   ci_low_adjusted_reliability |   ci_high_adjusted_reliability |   posterior_draws |
|:--------------------------|------------------------------:|------------------:|-------------------:|----------------------------:|----------------:|-----------------:|-------------------:|--------------------------:|---------------------------:|--------------------------------------------------:|------------------------:|---------------------------------------:|-------------------------------------:|---------------------------------:|-----------------------------------:|------------------------------:|-------------------------------:|------------------:|
| gpt-4.1-mini              |                      0.904568 |          0.89884  |           0.910021 |                    0.91626  |        0.907519 |         0.92422  |          0.0116918 |                0.00472799 |                  0.0182773 |                                            0.9996 |               0.0116918 |                              0.0116169 |                           0.00344761 |                        0.017177  |                           0.894054 |                      0.885346 |                       0.902402 |             10000 |
| qwen2.5:7b-instruct       |                      0.819356 |          0.810237 |           0.828072 |                    0.838463 |        0.824604 |         0.851469 |          0.0191066 |                0.00780196 |                  0.0295007 |                                            0.9993 |               0.0191066 |                              0.0189912 |                           0.00549695 |                        0.0277781 |                           0.803849 |                      0.790945 |                       0.816619 |             10000 |
| claude-haiku-4-5-20251001 |                      0.772936 |          0.76707  |           0.778707 |                    0.813784 |        0.806427 |         0.820959 |          0.0408481 |                0.0353595  |                  0.0461313 |                                            1      |               0.0408481 |                              0.0408035 |                           0.00275229 |                        0.0453398 |                           0.741999 |                      0.734668 |                       0.74937  |             10000 |
| llama3.1:8b               |                      0.647005 |          0.634002 |           0.659877 |                    0.688447 |        0.667918 |         0.708164 |          0.0414426 |                0.0253046  |                  0.0572221 |                                            1      |               0.0414426 |                              0.0413859 |                           0.00808879 |                        0.0547415 |                           0.620739 |                      0.604404 |                       0.637188 |             10000 |

## Multiple-Choice Output Quality

| model                     | condition   | source    |   value |    n |
|:--------------------------|:------------|:----------|--------:|-----:|
| claude-haiku-4-5-20251001 | baseline    | AdvGLUE   |   1     |  738 |
| claude-haiku-4-5-20251001 | baseline    | BBQ       |   1     | 2000 |
| claude-haiku-4-5-20251001 | baseline    | StereoSet |   1     | 1000 |
| claude-haiku-4-5-20251001 | safety      | AdvGLUE   |   1     |  738 |
| claude-haiku-4-5-20251001 | safety      | BBQ       |   1     | 2000 |
| claude-haiku-4-5-20251001 | safety      | StereoSet |   1     | 1000 |
| gpt-4.1-mini              | baseline    | AdvGLUE   |   1     |  738 |
| gpt-4.1-mini              | baseline    | BBQ       |   1     | 2000 |
| gpt-4.1-mini              | baseline    | StereoSet |   0.999 | 1000 |
| gpt-4.1-mini              | safety      | AdvGLUE   |   1     |  738 |
| gpt-4.1-mini              | safety      | BBQ       |   1     | 2000 |
| gpt-4.1-mini              | safety      | StereoSet |   1     | 1000 |
| llama3.1:8b               | baseline    | AdvGLUE   |   1     |  738 |
| llama3.1:8b               | baseline    | BBQ       |   1     | 2000 |
| llama3.1:8b               | baseline    | StereoSet |   1     | 1000 |
| llama3.1:8b               | safety      | AdvGLUE   |   1     |  738 |
| llama3.1:8b               | safety      | BBQ       |   1     | 2000 |
| llama3.1:8b               | safety      | StereoSet |   1     | 1000 |
| qwen2.5:7b-instruct       | baseline    | AdvGLUE   |   1     |  738 |
| qwen2.5:7b-instruct       | baseline    | BBQ       |   1     | 2000 |
| qwen2.5:7b-instruct       | baseline    | StereoSet |   1     | 1000 |
| qwen2.5:7b-instruct       | safety      | AdvGLUE   |   1     |  738 |
| qwen2.5:7b-instruct       | safety      | BBQ       |   1     | 2000 |
| qwen2.5:7b-instruct       | safety      | StereoSet |   1     | 1000 |

## BBQ bias-score Sample Sizes

| model                     | condition   | metric              |      value |   n_total |   n_non_unknown |
|:--------------------------|:------------|:--------------------|-----------:|----------:|----------------:|
| claude-haiku-4-5-20251001 | baseline    | bias_score_ambig    | 0.005      |      1000 |              17 |
| claude-haiku-4-5-20251001 | baseline    | bias_score_disambig | 0.0389222  |      1000 |             668 |
| claude-haiku-4-5-20251001 | safety      | bias_score_ambig    | 0.001      |      1000 |               1 |
| claude-haiku-4-5-20251001 | safety      | bias_score_disambig | 0.00666667 |      1000 |             600 |
| gpt-4.1-mini              | baseline    | bias_score_ambig    | 0.001      |      1000 |              47 |
| gpt-4.1-mini              | baseline    | bias_score_disambig | 0.0305085  |      1000 |             885 |
| gpt-4.1-mini              | safety      | bias_score_ambig    | 0.002      |      1000 |               8 |
| gpt-4.1-mini              | safety      | bias_score_disambig | 0.0219146  |      1000 |             867 |
| llama3.1:8b               | baseline    | bias_score_ambig    | 0.109      |      1000 |             705 |
| llama3.1:8b               | baseline    | bias_score_disambig | 0.0285111  |      1000 |             947 |
| llama3.1:8b               | safety      | bias_score_ambig    | 0.047      |      1000 |             399 |
| llama3.1:8b               | safety      | bias_score_disambig | 0.0147783  |      1000 |             812 |
| qwen2.5:7b-instruct       | baseline    | bias_score_ambig    | 0.034      |      1000 |             188 |
| qwen2.5:7b-instruct       | baseline    | bias_score_disambig | 0.00248756 |      1000 |             804 |
| qwen2.5:7b-instruct       | safety      | bias_score_ambig    | 0.022      |      1000 |              54 |
| qwen2.5:7b-instruct       | safety      | bias_score_disambig | 0.00156986 |      1000 |             637 |

## Generation Stability under the BOLD and RTP Stochastic Conditions

- For BOLD and RTP, continuations of up to 50 tokens were generated.
- The stochastic condition used `temperature=0.7` with three repetitions.
- `mean_within_item_generation_sd` is the mean, across items, of the standard deviation of repeated generation scores for the same item. Lower values indicate greater stability of the generated outputs.
- Termination due to `length` or `max_tokens` indicates that the specified continuation-length limit was reached and was not, in itself, treated as a generation failure.

| model                     | condition   | source   |     value |    n |
|:--------------------------|:------------|:---------|----------:|-----:|
| claude-haiku-4-5-20251001 | baseline    | BOLD     | 0.0961288 | 2000 |
| claude-haiku-4-5-20251001 | baseline    | RTP      | 0.0796743 | 2000 |
| claude-haiku-4-5-20251001 | safety      | BOLD     | 0.0652406 | 2000 |
| claude-haiku-4-5-20251001 | safety      | RTP      | 0.069282  | 2000 |
| gpt-4.1-mini              | baseline    | BOLD     | 0.0695707 | 2000 |
| gpt-4.1-mini              | baseline    | RTP      | 0.0352184 | 2000 |
| gpt-4.1-mini              | safety      | BOLD     | 0.0542709 | 2000 |
| gpt-4.1-mini              | safety      | RTP      | 0.0314656 | 2000 |
| llama3.1:8b               | baseline    | BOLD     | 0.144338  | 2000 |
| llama3.1:8b               | baseline    | RTP      | 0.0894893 | 2000 |
| llama3.1:8b               | safety      | BOLD     | 0.116913  | 2000 |
| llama3.1:8b               | safety      | RTP      | 0.101902  | 2000 |
| qwen2.5:7b-instruct       | baseline    | BOLD     | 0.0776536 | 2000 |
| qwen2.5:7b-instruct       | baseline    | RTP      | 0.0484974 | 2000 |
| qwen2.5:7b-instruct       | safety      | BOLD     | 0.066684  | 2000 |
| qwen2.5:7b-instruct       | safety      | RTP      | 0.0352184 | 2000 |

## Notes on Interpretation

- Latent reliability was estimated using a Rasch-type hierarchical logistic model that included fixed effects for model, condition, and benchmark, as well as random effects for item difficulty.
- The 95% approximate credible intervals for baseline reliability, safety-conditioned reliability, the signed condition effect, and sensitivity-adjusted reliability were based on Monte Carlo propagation of the mean-field variational Bayes posterior distribution of the fixed effects.
- `condition_effect` is the signed difference between the safety and baseline conditions. Its direction and uncertainty should be assessed by determining whether its credible interval includes zero.
- `condition_sensitivity` is the nonnegative penalty magnitude defined as `abs(condition_effect)`. The lower bound of the absolute-value distribution should not be used to test a null hypothesis. Instead, the posterior mean, standard deviation, and one-sided 95% upper bound are reported.
- For the BBQ bias-score rows, `n` and `n_total` denote the total number of items, whereas `n_non_unknown` denotes the number of non-UNKNOWN responses used to estimate the direction of the raw bias.
- Multiple-choice parsing failures were treated as `favorable=0` in the primary analysis and were separately tracked using `parse_valid` and `choice_parse_rate`.
- The bootstrap values are empirical sensitivity-adjusted composite values obtained by resampling item clusters within each benchmark. They are not the same estimand as the latent-model scores.
- Differences under the safety condition represent the effects of the safety-instruction intervention and should be distinguished from invariance to meaning-preserving prompt variations.