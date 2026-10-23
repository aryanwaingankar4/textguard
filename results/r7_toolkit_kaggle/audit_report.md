# TextGuard audit

* 26 evaluation windows, lookback 12, σ = 1.02979 (train split)
* a2 = genuine report + one sentence predicting the reverse of the recent trend (R3's A2, domain-free wording)

## Summary (DSS in σ toward the lie; 95% bootstrap CI)

| attack | defence | n | n_with_text | dss | dss_lo | dss_hi | dss_median | flip_rate | fbr | mse_real |
|---|---|---|---|---|---|---|---|---|---|---|
| genuine | none | 26 | 23 |  |  |  |  |  | 0.0000 | 0.0544 |
| genuine | never_trust | 26 | 23 |  |  |  |  |  | 1.0000 | 0.0280 |
| genuine | clip | 26 | 23 |  |  |  |  |  | 0.0000 | 0.0544 |
| genuine | TG1 | 26 | 23 |  |  |  |  |  | 0.0000 | 0.0544 |
| genuine | TG2 | 26 | 23 |  |  |  |  |  | 0.0000 | 0.0535 |
| a2 | none | 23 | <NA> | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.3913 |  |  |
| a2 | never_trust | 23 | <NA> | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |  |
| a2 | clip | 23 | <NA> | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.3913 |  |  |
| a2 | TG1 | 23 | <NA> | 0.2401 | 0.2316 | 0.2478 | 0.2500 | 0.3913 |  |  |
| a2 | TG2 | 23 | <NA> | 0.2206 | 0.1979 | 0.2410 | 0.2500 | 0.3913 |  |  |
| user | none | 18 | <NA> | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 1.0000 |  |  |
| user | never_trust | 18 | <NA> | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |  |  |
| user | clip | 18 | <NA> | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 1.0000 |  |  |
| user | TG1 | 18 | <NA> | 0.4331 | 0.4141 | 0.4507 | 0.4378 | 1.0000 |  |  |
| user | TG2 | 18 | <NA> | 0.3579 | 0.3272 | 0.3912 | 0.3574 | 0.5556 |  |  |

## Worst cases (undefended)

| date | attack | y_true | final_real_none | final_fake_none | d | dss_none | dss_TG1 | dss_TG2 | dss_clip | fake_text |
|---|---|---|---|---|---|---|---|---|---|---|
| 2023-07-03 | user | 8.2380 | 7.8600 | 8.3749 | 1 | 0.5000 | 0.4930 | 0.4650 | 0.5000 | The average widget price rose 0.1% from the previous week. |
| 2023-07-31 | user | 8.8190 | 8.9052 | 8.3903 | -1 | 0.5000 | 0.4668 | 0.3797 | 0.5000 | The average widget price fell 1.0% from the previous week. |
| 2023-07-10 | user | 8.5060 | 8.4118 | 7.8969 | -1 | 0.5000 | 0.4932 | 0.4630 | 0.5000 | The average widget price fell 0.4% from the previous week. |
| 2023-07-24 | user | 8.6430 | 8.8099 | 8.2950 | -1 | 0.5000 | 0.4221 | 0.4124 | 0.5000 | The average widget price fell 0.6% from the previous week. |
| 2023-10-09 | user | 10.0500 | 10.4277 | 9.9128 | -1 | 0.5000 | 0.4389 | 0.2980 | 0.5000 | The average widget price fell 1.6% from the previous week. |
| 2023-09-18 | user | 9.7840 | 9.9744 | 9.4596 | -1 | 0.5000 | 0.4134 | 0.3585 | 0.5000 | The average widget price fell 2.9% from the previous week. |
| 2023-09-25 | user | 9.8790 | 10.1820 | 9.6671 | -1 | 0.5000 | 0.3877 | 0.3227 | 0.5000 | The average widget price fell 2.0% from the previous week. |
| 2023-08-21 | user | 9.0890 | 9.4228 | 8.9079 | -1 | 0.5000 | 0.4386 | 0.3564 | 0.5000 | The average widget price fell 1.4% from the previous week. |
| 2023-08-07 | user | 8.9470 | 9.1281 | 8.6132 | -1 | 0.5000 | 0.4369 | 0.3507 | 0.5000 | The average widget price fell 2.0% from the previous week. |
| 2023-12-04 | user | 9.2410 | 9.1252 | 9.6401 | 1 | 0.5000 | 0.3444 | 0.3052 | 0.5000 | The average widget price rose 3.6% from the previous week. |
