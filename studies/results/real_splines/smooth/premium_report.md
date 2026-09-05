# Tweedie model of `avenue_pure_premium`

**Not usable as it stands** · plan `a293dfb03d14`

This model should not be used as it stands. Overall actual over expected is 1.4228 (19291839.8900 actual against 13558888.1750 expected). Investigate population differences, loss volatility and model specification before changing the rate level. It explains 11.2% of the deviance on the data it was measured against, with actual over expected at 1.4228 and a Gini of 0.488.

## Findings

| Severity | Stage | Finding |
|---|---|---|
| high | validation | Overall actual over expected is 1.4228 (19291839.8900 actual against 13558888.1750 expected). Investigate population differences, loss volatility and model specification before changing the rate level. |
| medium | validation | 9 of 10 equal-exposure buckets have actual over expected outside 10%: buckets 0, 1, 2, 3, 5, 6, 7, 8, 9. These are descriptive flags: review exposure, claim support and loss volatility before attributing the pattern to a missing interaction or a mis-specified band. |
| low | plan | Table 'age' is a continuous spline. Support intervals are not knot-parameter counts; knot inference uses the continuous basis; pre-fit spline conditioning is not yet available. |
| low | plan | Table 'vehicle_age' is a continuous spline. Support intervals are not knot-parameter counts; knot inference uses the continuous basis; pre-fit spline conditioning is not yet available. |
| low | plan | Table 'bonus' is a continuous spline. Support intervals are not knot-parameter counts; knot inference uses the continuous basis; pre-fit spline conditioning is not yet available. |
| low | validation | 'age': A/E rows describe the displayed support intervals, not individual knot parameters. Interval counts do not establish whether a knot value is estimable. |
| low | validation | 'vehicle_age': A/E rows describe the displayed support intervals, not individual knot parameters. Interval counts do not establish whether a knot value is estimable. |
| low | validation | 'bonus': A/E rows describe the displayed support intervals, not individual knot parameters. Interval counts do not establish whether a knot value is estimable. |
| low | validation | 1 table rows hold less than 0.10% of total exposure each. Their actual-versus-expected figures are noisy and should not be read as evidence on their own. |

## The model

| Term | Kind | Rows | Parameters | Base |
|---|---|---:|---:|---|
| intercept | intercept | 1 | 1 | — |
| age | natural_cubic_spline | 5 | 4 | first knot |
| vehicle_age | natural_cubic_spline | 5 | 4 | first knot |
| bonus | natural_cubic_spline | 3 | 2 | first knot |
| region | categorical | 22 | 21 | R11 |
| fuel | categorical | 2 | 1 | 'Diesel' |

## Fit

| | |
|---|---|
| Converged | yes after 103 sweeps (score 9.53e-12) |
| Deviance | 20780057.098548 against a null of 22099342.683457 |
| Pseudo R-squared | 0.0597 |
| Parameters | 33 |
| Covariance | model_based |
| Dispersion | 6966.067008 |

## Validation

| | |
|---|---|
| Rows | 169504 scored of 169504 |
| Actual / expected | 1.4228 |
| Gini | 0.4883 |
| Lift, top over bottom | 10.96x |
| Out-of-sample pseudo R-squared | 0.1120 |

### Calibration, by equal-exposure bucket

| bin | n | weight | mean_predicted | actual | expected | actual_rate | expected_rate | ae_ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 15225 | 8974.3588 | 59.1007 | 773244.4400 | 530391.0553 | 86.1615 | 59.1007 | 1.4579 |
| 1 | 15082 | 8971.1362 | 76.4265 | 526116.9700 | 685632.2261 | 58.6455 | 76.4265 | 0.7673 |
| 2 | 15361 | 8974.4716 | 87.8137 | 893332.8300 | 788081.8803 | 99.5416 | 87.8137 | 1.1336 |
| 3 | 15475 | 8971.7422 | 98.0124 | 1045881.4700 | 879342.3010 | 116.5751 | 98.0124 | 1.1894 |
| 4 | 15860 | 8974.3606 | 108.0572 | 1016030.9900 | 969744.5871 | 113.2149 | 108.0572 | 1.0477 |
| 5 | 16306 | 8969.7433 | 119.6349 | 838171.0500 | 1073093.9471 | 93.4443 | 119.6349 | 0.7811 |
| 6 | 16847 | 8972.0744 | 134.9484 | 1349017.6900 | 1210767.0606 | 150.3574 | 134.9484 | 1.1142 |
| 7 | 17977 | 8972.4869 | 160.1873 | 1838736.0000 | 1437278.4872 | 204.9305 | 160.1873 | 1.2793 |
| 8 | 19597 | 8972.6782 | 215.5060 | 2535188.5100 | 1933665.6132 | 282.5454 | 215.5060 | 1.3111 |
| 9 | 21774 | 8972.2019 | 451.4935 | 8476119.9400 | 4050891.0170 | 944.7090 | 451.4935 | 2.0924 |

### Actual versus expected

**age**

| Spline_Feature | Support_Lower | Support_Upper | Lower_Inclusive | Upper_Inclusive | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|---|---|---|
| age | -inf | 18.0000 | false | true | 191 | 49.8864 | 80999.8200 | 35825.0586 | 2.2610 |
| age | 18.0000 | 34.0000 | false | true | 42595 | 19187.1580 | 9096162.3100 | 4047584.0665 | 2.2473 |
| age | 34.0000 | 44.0000 | false | true | 42488 | 21859.6967 | 3867799.4500 | 2784270.1809 | 1.3892 |
| age | 44.0000 | 55.0000 | false | true | 44804 | 24608.4405 | 3309510.4500 | 3470688.8739 | 0.9536 |
| age | 55.0000 | inf | false | false | 39426 | 24020.0726 | 2937367.8600 | 3220519.9950 | 0.9121 |

**vehicle_age**

| Spline_Feature | Support_Lower | Support_Upper | Lower_Inclusive | Upper_Inclusive | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|---|---|---|
| vehicle_age | -inf | 0.0000 | false | true | 14198 | 4133.5166 | 550198.4900 | 658762.2093 | 0.8352 |
| vehicle_age | 0.0000 | 2.0000 | false | true | 32796 | 15754.2959 | 2089110.2600 | 2609303.2667 | 0.8006 |
| vehicle_age | 2.0000 | 6.0000 | false | true | 41858 | 23153.3951 | 3023082.5300 | 3780481.0988 | 0.7997 |
| vehicle_age | 6.0000 | 11.0000 | false | true | 41337 | 23928.1031 | 4911889.3500 | 3597005.8892 | 1.3655 |
| vehicle_age | 11.0000 | inf | false | false | 39315 | 22755.9435 | 8717559.2600 | 2913335.7111 | 2.9923 |

**bonus**

| Spline_Feature | Support_Lower | Support_Upper | Lower_Inclusive | Upper_Inclusive | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|---|---|---|
| bonus | -inf | 50.0000 | false | true | 96090 | 56322.3928 | 5575140.9300 | 6000976.1372 | 0.9290 |
| bonus | 50.0000 | 64.0000 | false | true | 31177 | 15618.6506 | 3304956.4800 | 1941026.9620 | 1.7027 |
| bonus | 64.0000 | inf | false | false | 42237 | 17784.2108 | 10411742.4800 | 5616885.0758 | 1.8537 |

**region**

| region | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio | region_Level |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 17466 | 7566.3725 | 896161.0100 | 1250019.7652 | 0.7169 | R11 |
| 1 | 1.4536 | 788 | 302.7738 | 32707.7500 | 197349.9049 | 0.1657 | R21 |
| 2 | -0.1191 | 1979 | 887.7460 | 289324.4000 | 137407.2460 | 2.1056 | R22 |
| 3 | -0.1747 | 2157 | 769.9523 | 57783.2300 | 92875.6349 | 0.6222 | R23 |
| 4 | -0.0652 | 40207 | 25735.7253 | 8778694.1200 | 3345658.6580 | 2.6239 | R24 |
| 5 | -0.1489 | 2708 | 1662.8860 | 354281.1500 | 206613.3159 | 1.7147 | R25 |
| 6 | -0.2207 | 2556 | 1232.4323 | 180691.3000 | 157512.5488 | 1.1472 | R26 |
| 7 | -0.0733 | 6778 | 2889.2489 | 427194.0800 | 448568.7846 | 0.9523 | R31 |
| 8 | -0.2473 | 3245 | 2020.8586 | 425235.8100 | 205997.8861 | 2.0643 | R41 |
| 9 | -0.4873 | 571 | 312.4010 | 34115.8500 | 29743.2177 | 1.1470 | R42 |
| 10 | -0.0799 | 332 | 134.6810 | 8823.9500 | 20815.1515 | 0.4239 | R43 |
| 11 | -0.2029 | 9791 | 5518.0675 | 637352.1500 | 630259.5755 | 1.0113 | R52 |
| 12 | -0.0053 | 10556 | 6932.1967 | 927553.8700 | 943364.5988 | 0.9832 | R53 |
| 13 | -0.1664 | 4752 | 2748.1134 | 313067.3300 | 348878.7610 | 0.8974 | R54 |
| 14 | -0.0435 | 7767 | 3570.3315 | 467487.4500 | 509214.6065 | 0.9181 | R72 |
| 15 | -0.5728 | 4383 | 1820.8117 | 128607.6300 | 148578.8601 | 0.8656 | R73 |
| 16 | -0.1769 | 1131 | 609.9538 | 85518.3500 | 67168.9988 | 1.2732 | R74 |
| 17 | 0.3099 | 21080 | 11281.2383 | 2790694.5100 | 2283126.7391 | 1.2223 | R82 |
| 18 | -0.5559 | 1304 | 583.9605 | 36463.6700 | 51022.2814 | 0.7147 | R83 |
| 19 | 0.0005 | 8956 | 3723.0506 | 439727.3400 | 558763.0855 | 0.7870 | R91 |
| 20 | 0.2989 | 19846 | 8968.5326 | 1910384.5200 | 1792754.5569 | 1.0656 | R93 |
| 21 | 0.5511 | 1151 | 453.9201 | 69970.4200 | 133193.9980 | 0.5253 | R94 |

**fuel**

| fuel | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio | fuel_Level |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 83146 | 42910.0428 | 7378460.2600 | 6971109.7926 | 1.0584 | 'Diesel' |
| 1 | -0.2017 | 86358 | 46815.2113 | 11913379.6300 | 6587778.3824 | 1.8084 | 'Regular' |

## Rating tables

**intercept**

| Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|
| 5.2454 | 5.2454 | 0.4012 | estimated | 189.6910 |

**age**

| age | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 18.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 34.0000 | -0.9536 | -0.9536 | 0.3457 | estimated | 0.3853 |
| 44.0000 | -0.4447 | -0.4447 | 0.3119 | estimated | 0.6410 |
| 55.0000 | -0.5031 | -0.5031 | 0.3195 | estimated | 0.6046 |
| 100.0000 | 0.9783 | 0.9783 | 0.6592 | estimated | 2.6600 |

**vehicle_age**

| vehicle_age | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 0.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 2.0000 | 0.1224 | 0.1224 | 0.1863 | estimated | 1.1302 |
| 6.0000 | 0.0521 | 0.0521 | 0.1951 | estimated | 1.0534 |
| 11.0000 | -0.0952 | -0.0952 | 0.1823 | estimated | 0.9092 |
| 100.0000 | -2.5603 | -2.5603 | 6.6381 | estimated | 0.0773 |

**bonus**

| bonus | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 50.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 64.0000 | 0.5518 | 0.5518 | 0.0853 | estimated | 1.7365 |
| 230.0000 | 2.8087 | 2.8087 | 2.0819 | estimated | 16.5883 |

**region**

| region | Rating_Factor | region_Level | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | R11 | 0.0000 | 0.0000 | reference | 1.0000 |
| 1 | 1.4536 | R21 | 1.4536 | 0.5777 | estimated | 4.2787 |
| 2 | -0.1191 | R22 | -0.1191 | 0.4966 | estimated | 0.8877 |
| 3 | -0.1747 | R23 | -0.1747 | 0.5448 | estimated | 0.8397 |
| 4 | -0.0652 | R24 | -0.0652 | 0.1890 | estimated | 0.9369 |
| 5 | -0.1489 | R25 | -0.1489 | 0.3958 | estimated | 0.8616 |
| 6 | -0.2207 | R26 | -0.2207 | 0.4452 | estimated | 0.8020 |
| 7 | -0.0733 | R31 | -0.0733 | 0.3064 | estimated | 0.9293 |
| 8 | -0.2473 | R41 | -0.2473 | 0.3791 | estimated | 0.7809 |
| 9 | -0.4873 | R42 | -0.4873 | 0.9340 | estimated | 0.6143 |
| 10 | -0.0799 | R43 | -0.0799 | 1.1723 | estimated | 0.9232 |
| 11 | -0.2029 | R52 | -0.2029 | 0.2607 | estimated | 0.8164 |
| 12 | -0.0053 | R53 | -0.0053 | 0.2380 | estimated | 0.9948 |
| 13 | -0.1664 | R54 | -0.1664 | 0.3245 | estimated | 0.8467 |
| 14 | -0.0435 | R72 | -0.0435 | 0.2876 | estimated | 0.9574 |
| 15 | -0.5728 | R73 | -0.5728 | 0.4206 | estimated | 0.5640 |
| 16 | -0.1769 | R74 | -0.1769 | 0.6362 | estimated | 0.8379 |
| 17 | 0.3099 | R82 | 0.3099 | 0.2024 | estimated | 1.3633 |
| 18 | -0.5559 | R83 | -0.5559 | 0.6888 | estimated | 0.5735 |
| 19 | 0.0005 | R91 | 0.0005 | 0.2802 | estimated | 1.0005 |
| 20 | 0.2989 | R93 | 0.2989 | 0.2114 | estimated | 1.3484 |
| 21 | 0.5511 | R94 | 0.5511 | 0.5912 | estimated | 1.7352 |

**fuel**

| fuel | Rating_Factor | fuel_Level | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | 'Diesel' | 0.0000 | 0.0000 | reference | 1.0000 |
| 1 | -0.2017 | 'Regular' | -0.2017 | 0.0968 | estimated | 0.8173 |

## Plan

The model's source code. Save it, edit it, re-run it.

```json
{
  "family": "tweedie",
  "tweedie_power": 1.5,
  "exposure": "exposure",
  "exposure_role": "weight",
  "terms": [
    {
      "kind": "spline",
      "column": "age",
      "knots": {
        "kind": "quantile",
        "n": 5
      }
    },
    {
      "kind": "spline",
      "column": "vehicle_age",
      "knots": {
        "kind": "quantile",
        "n": 5
      }
    },
    {
      "kind": "spline",
      "column": "bonus",
      "knots": {
        "kind": "quantile",
        "n": 5
      }
    },
    {
      "kind": "categorical",
      "column": "region",
      "base": {
        "kind": "first"
      }
    },
    {
      "kind": "categorical",
      "column": "fuel",
      "base": {
        "kind": "first"
      }
    }
  ]
}
```
