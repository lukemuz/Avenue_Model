# Gamma model of `avenue_severity`

**Not usable as it stands** · plan `6b76e42d269a`

This model should not be used as it stands. Overall actual over expected is 1.4384 (19291839.8900 actual against 13412030.9025 expected). Investigate population differences, loss volatility and model specification before changing the rate level. It explains 7.2% of the deviance on the data it was measured against, with actual over expected at 1.4384 and a Gini of 0.258.

## Findings

| Severity | Stage | Finding |
|---|---|---|
| high | validation | Overall actual over expected is 1.4384 (19291839.8900 actual against 13412030.9025 expected). Investigate population differences, loss volatility and model specification before changing the rate level. |
| medium | validation | 9 of 10 equal-exposure buckets have actual over expected outside 10%: buckets 0, 1, 2, 3, 4, 5, 6, 8, 9. These are descriptive flags: review exposure, claim support and loss volatility before attributing the pattern to a missing interaction or a mis-specified band. |
| low | plan | Table 'age' is a continuous spline. Support intervals are not knot-parameter counts; knot inference uses the continuous basis; pre-fit spline conditioning is not yet available. |
| low | plan | Table 'vehicle_age' is a continuous spline. Support intervals are not knot-parameter counts; knot inference uses the continuous basis; pre-fit spline conditioning is not yet available. |
| low | plan | Table 'bonus' is a continuous spline. Support intervals are not knot-parameter counts; knot inference uses the continuous basis; pre-fit spline conditioning is not yet available. |
| low | validation | 'age': A/E rows describe the displayed support intervals, not individual knot parameters. Interval counts do not establish whether a knot value is estimable. |
| low | validation | 'vehicle_age': A/E rows describe the displayed support intervals, not individual knot parameters. Interval counts do not establish whether a knot value is estimable. |
| low | validation | 'bonus': A/E rows describe the displayed support intervals, not individual knot parameters. Interval counts do not establish whether a knot value is estimable. |

## The model

| Term | Kind | Rows | Parameters | Base |
|---|---|---:|---:|---|
| intercept | intercept | 1 | 1 | — |
| age | natural_cubic_spline | 5 | 4 | first knot |
| vehicle_age | natural_cubic_spline | 5 | 4 | first knot |
| bonus | natural_cubic_spline | 4 | 3 | first knot |
| region | categorical | 22 | 21 | R11 |
| fuel | categorical | 2 | 1 | 'Diesel' |

## Fit

| | |
|---|---|
| Converged | yes after 26 sweeps (score 4.80e-12) |
| Deviance | 28251.558069 against a null of 29689.614243 |
| Pseudo R-squared | 0.0484 |
| Parameters | 34 |
| Covariance | model_based |
| Dispersion | 18.319254 |
| AIC | 373759.0925 |
| BIC | 374025.6421 |

## Validation

| | |
|---|---|
| Rows | 6180 scored of 6180 |
| Actual / expected | 1.4384 |
| Gini | 0.2580 |
| Lift, top over bottom | 5.63x |
| Out-of-sample pseudo R-squared | 0.0715 |

### Calibration, by equal-exposure bucket

| bin | n | weight | mean_predicted | actual | expected | actual_rate | expected_rate | ae_ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 629 | 657.0000 | 1391.2321 | 1255944.2400 | 914039.5132 | 1911.6351 | 1391.2321 | 1.3741 |
| 1 | 632 | 657.0000 | 1595.1499 | 2034763.5900 | 1048013.5078 | 3097.0526 | 1595.1499 | 1.9415 |
| 2 | 609 | 656.0000 | 1705.0545 | 1287052.1500 | 1118515.7834 | 1961.9697 | 1705.0545 | 1.1507 |
| 3 | 613 | 657.0000 | 1785.0032 | 1715258.8700 | 1172747.0851 | 2610.7441 | 1785.0032 | 1.4626 |
| 4 | 619 | 656.0000 | 1869.1055 | 1101245.7400 | 1226133.2376 | 1678.7283 | 1869.1055 | 0.8981 |
| 5 | 622 | 657.0000 | 1970.1836 | 1157130.7600 | 1294410.6047 | 1761.2340 | 1970.1836 | 0.8939 |
| 6 | 629 | 657.0000 | 2079.2449 | 1074469.0800 | 1366063.8670 | 1635.4172 | 2079.2449 | 0.7865 |
| 7 | 616 | 656.0000 | 2210.3885 | 1453420.8500 | 1450014.8356 | 2215.5806 | 2210.3885 | 1.0023 |
| 8 | 613 | 660.0000 | 2449.9916 | 1181106.1500 | 1616994.4531 | 1789.5548 | 2449.9916 | 0.7304 |
| 9 | 598 | 653.0000 | 3376.8729 | 7031448.4600 | 2205098.0150 | 10767.9149 | 3376.8729 | 3.1887 |

### Actual versus expected

**age**

| Spline_Feature | Support_Lower | Support_Upper | Lower_Inclusive | Upper_Inclusive | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|---|---|---|
| age | -inf | 18.0000 | false | true | 11 | 11.0000 | 80999.8200 | 41462.6930 | 1.9536 |
| age | 18.0000 | 34.0000 | false | true | 1626 | 1730.0000 | 9096162.3100 | 4065231.2050 | 2.2376 |
| age | 34.0000 | 45.0000 | false | true | 1618 | 1704.0000 | 4078846.4800 | 3046608.3637 | 1.3388 |
| age | 45.0000 | 54.0000 | false | true | 1395 | 1500.0000 | 2898379.8500 | 2794872.2481 | 1.0370 |
| age | 54.0000 | inf | false | false | 1530 | 1621.0000 | 3137451.4300 | 3463856.3926 | 0.9058 |

**vehicle_age**

| Spline_Feature | Support_Lower | Support_Upper | Lower_Inclusive | Upper_Inclusive | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|---|---|---|
| vehicle_age | -inf | 0.0000 | false | true | 265 | 285.0000 | 550198.4900 | 601410.0740 | 0.9148 |
| vehicle_age | 0.0000 | 3.0000 | false | true | 1528 | 1642.0000 | 2869199.8000 | 3644498.7188 | 0.7873 |
| vehicle_age | 3.0000 | 7.0000 | false | true | 1602 | 1711.0000 | 4111588.9400 | 3436490.7425 | 1.1964 |
| vehicle_age | 7.0000 | 11.0000 | false | true | 1411 | 1478.0000 | 3043293.4000 | 2737622.0400 | 1.1117 |
| vehicle_age | 11.0000 | inf | false | false | 1374 | 1450.0000 | 8717559.2600 | 2992009.3271 | 2.9136 |

**bonus**

| Spline_Feature | Support_Lower | Support_Upper | Lower_Inclusive | Upper_Inclusive | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|---|---|---|
| bonus | -inf | 50.0000 | false | true | 2660 | 2806.0000 | 5575140.9300 | 5515481.9998 | 1.0108 |
| bonus | 50.0000 | 55.0000 | false | true | 392 | 412.0000 | 666382.8300 | 762167.8889 | 0.8743 |
| bonus | 55.0000 | 76.0000 | false | true | 1704 | 1802.0000 | 4457103.3300 | 3399738.1585 | 1.3110 |
| bonus | 76.0000 | inf | false | false | 1424 | 1546.0000 | 8593212.8000 | 3734642.8553 | 2.3009 |

**region**

| region | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio | region_Level |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 574 | 616.0000 | 896161.0100 | 1134085.1801 | 0.7902 | R11 |
| 1 | 1.5014 | 18 | 20.0000 | 32707.7500 | 170122.8723 | 0.1923 | R21 |
| 2 | -0.0987 | 74 | 75.0000 | 289324.4000 | 126612.5316 | 2.2851 | R22 |
| 3 | -0.1352 | 36 | 40.0000 | 57783.2300 | 66389.8130 | 0.8704 | R23 |
| 4 | 0.1372 | 1542 | 1598.0000 | 8778694.1200 | 3421971.7818 | 2.5654 | R24 |
| 5 | -0.0516 | 102 | 110.0000 | 354281.1500 | 193724.4066 | 1.8288 | R25 |
| 6 | 0.0183 | 81 | 89.0000 | 180691.3000 | 172292.0747 | 1.0487 | R26 |
| 7 | -0.0177 | 232 | 254.0000 | 427194.0800 | 468841.7595 | 0.9112 | R31 |
| 8 | 0.0109 | 124 | 130.0000 | 425235.8100 | 242264.7862 | 1.7553 | R41 |
| 9 | -0.3775 | 21 | 24.0000 | 34115.8500 | 30420.7538 | 1.1215 | R42 |
| 10 | 0.1217 | 7 | 7.0000 | 8823.9500 | 14328.4366 | 0.6158 | R43 |
| 11 | -0.0996 | 362 | 381.0000 | 637352.1500 | 618651.4005 | 1.0302 | R52 |
| 12 | 0.1332 | 451 | 467.0000 | 927553.8700 | 974232.2720 | 0.9521 | R53 |
| 13 | -0.0502 | 208 | 215.0000 | 313067.3300 | 391666.6007 | 0.7993 | R54 |
| 14 | -0.0119 | 217 | 236.0000 | 467487.4500 | 435396.9351 | 1.0737 | R72 |
| 15 | -0.1488 | 99 | 104.0000 | 128607.6300 | 162125.0731 | 0.7933 | R73 |
| 16 | -0.2903 | 41 | 44.0000 | 85518.3500 | 57830.5329 | 1.4788 | R74 |
| 17 | 0.1694 | 974 | 1030.0000 | 2790694.5100 | 2229877.4480 | 1.2515 | R82 |
| 18 | -0.3224 | 30 | 30.0000 | 36463.6700 | 40142.5132 | 0.9084 | R83 |
| 19 | 0.0899 | 232 | 260.0000 | 439727.3400 | 507219.6780 | 0.8669 | R91 |
| 20 | 0.2237 | 725 | 801.0000 | 1910384.5200 | 1814469.8851 | 1.0529 | R93 |
| 21 | 0.7426 | 30 | 35.0000 | 69970.4200 | 139364.1675 | 0.5021 | R94 |

**fuel**

| fuel | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio | fuel_Level |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 3162 | 3359.0000 | 7378460.2600 | 6830761.3218 | 1.0802 | 'Diesel' |
| 1 | -0.0236 | 3018 | 3207.0000 | 11913379.6300 | 6581269.5806 | 1.8102 | 'Regular' |

## Rating tables

**intercept**

| Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|
| 8.1289 | 8.1289 | 0.2354 | estimated | 3391.0429 |

**age**

| age | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 18.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 34.0000 | -0.7009 | -0.7009 | 0.2039 | estimated | 0.4961 |
| 45.0000 | -0.6585 | -0.6585 | 0.1818 | estimated | 0.5176 |
| 54.0000 | -0.6601 | -0.6601 | 0.1862 | estimated | 0.5168 |
| 99.0000 | 0.3715 | 0.3715 | 0.4433 | estimated | 1.4499 |

**vehicle_age**

| vehicle_age | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 0.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 3.0000 | 0.0743 | 0.0743 | 0.1332 | estimated | 1.0771 |
| 7.0000 | -0.1368 | -0.1368 | 0.1197 | estimated | 0.8721 |
| 11.0000 | -0.1241 | -0.1241 | 0.1127 | estimated | 0.8833 |
| 99.0000 | -0.5791 | -0.5791 | 3.2825 | estimated | 0.5604 |

**bonus**

| bonus | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 50.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 55.0000 | -0.0451 | -0.0451 | 0.0513 | estimated | 0.9559 |
| 76.0000 | 0.0341 | 0.0341 | 0.0813 | estimated | 1.0347 |
| 208.0000 | -0.7791 | -0.7791 | 0.8877 | estimated | 0.4588 |

**region**

| region | Rating_Factor | region_Level | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | R11 | 0.0000 | 0.0000 | reference | 1.0000 |
| 1 | 1.5014 | R21 | 1.5014 | 0.5753 | estimated | 4.4878 |
| 2 | -0.0987 | R22 | -0.0987 | 0.2937 | estimated | 0.9060 |
| 3 | -0.1352 | R23 | -0.1352 | 0.3338 | estimated | 0.8735 |
| 4 | 0.1372 | R24 | 0.1372 | 0.1170 | estimated | 1.1470 |
| 5 | -0.0516 | R25 | -0.0516 | 0.2519 | estimated | 0.9497 |
| 6 | 0.0183 | R26 | 0.0183 | 0.2851 | estimated | 1.0185 |
| 7 | -0.0177 | R31 | -0.0177 | 0.1902 | estimated | 0.9825 |
| 8 | 0.0109 | R41 | 0.0109 | 0.2532 | estimated | 1.0110 |
| 9 | -0.3775 | R42 | -0.3775 | 0.5284 | estimated | 0.6856 |
| 10 | 0.1217 | R43 | 0.1217 | 0.7751 | estimated | 1.1294 |
| 11 | -0.0996 | R52 | -0.0996 | 0.1584 | estimated | 0.9052 |
| 12 | 0.1332 | R53 | 0.1332 | 0.1511 | estimated | 1.1425 |
| 13 | -0.0502 | R54 | -0.0502 | 0.2026 | estimated | 0.9510 |
| 14 | -0.0119 | R72 | -0.0119 | 0.1788 | estimated | 0.9881 |
| 15 | -0.1488 | R73 | -0.1488 | 0.2804 | estimated | 0.8617 |
| 16 | -0.2903 | R74 | -0.2903 | 0.3599 | estimated | 0.7480 |
| 17 | 0.1694 | R82 | 0.1694 | 0.1237 | estimated | 1.1846 |
| 18 | -0.3224 | R83 | -0.3224 | 0.4181 | estimated | 0.7244 |
| 19 | 0.0899 | R91 | 0.0899 | 0.1794 | estimated | 1.0940 |
| 20 | 0.2237 | R93 | 0.2237 | 0.1335 | estimated | 1.2508 |
| 21 | 0.7426 | R94 | 0.7426 | 0.4455 | estimated | 2.1014 |

**fuel**

| fuel | Rating_Factor | fuel_Level | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | 'Diesel' | 0.0000 | 0.0000 | reference | 1.0000 |
| 1 | -0.0236 | 'Regular' | -0.0236 | 0.0624 | estimated | 0.9767 |

## Plan

The model's source code. Save it, edit it, re-run it.

```json
{
  "family": "gamma",
  "tweedie_power": 1.5,
  "exposure": "paid_claims",
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
