# Poisson model of `avenue_frequency`

**Usable with caveats** · plan `4db536baa60c`

It explains 4.1% of the deviance on the data it was measured against, with actual over expected at 0.9905 and a Gini of 0.277. It is usable, with 8 caveats to carry forward — the first is: 3 of 10 equal-exposure buckets have actual over expected outside 10%: buckets 0, 4, 8. These are descriptive flags: review exposure, claim support and loss volatility before attributing the pattern to a missing interaction or a mis-specified band.

## Findings

| Severity | Stage | Finding |
|---|---|---|
| medium | validation | 3 of 10 equal-exposure buckets have actual over expected outside 10%: buckets 0, 4, 8. These are descriptive flags: review exposure, claim support and loss volatility before attributing the pattern to a missing interaction or a mis-specified band. |
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
| Converged | yes after 42 sweeps (score 4.31e-12) |
| Deviance | 123874.988280 against a null of 129197.537340 |
| Pseudo R-squared | 0.0412 |
| Parameters | 33 |
| Covariance | model_based |
| Dispersion | 1.000000 |
| AIC | 153173.0619 |
| BIC | 153540.6568 |

## Validation

| | |
|---|---|
| Rows | 169504 scored of 169504 |
| Actual / expected | 0.9905 |
| Gini | 0.2767 |
| Lift, top over bottom | 5.46x |
| Out-of-sample pseudo R-squared | 0.0407 |

### Calibration, by equal-exposure bucket

| bin | n | weight | mean_predicted | actual | expected | actual_rate | expected_rate | ae_ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 14933 | 8973.3407 | 0.0368 | 275.0000 | 330.4875 | 0.0306 | 0.0368 | 0.8321 |
| 1 | 15223 | 8971.9644 | 0.0455 | 389.0000 | 407.9557 | 0.0434 | 0.0455 | 0.9535 |
| 2 | 15294 | 8975.5007 | 0.0505 | 433.0000 | 453.1630 | 0.0482 | 0.0505 | 0.9555 |
| 3 | 15798 | 8969.8018 | 0.0546 | 485.0000 | 490.1641 | 0.0541 | 0.0546 | 0.9895 |
| 4 | 16077 | 8972.0473 | 0.0588 | 470.0000 | 527.8858 | 0.0524 | 0.0588 | 0.8903 |
| 5 | 16478 | 8972.5028 | 0.0632 | 593.0000 | 567.1112 | 0.0661 | 0.0632 | 1.0457 |
| 6 | 16625 | 8976.1830 | 0.0688 | 635.0000 | 617.9662 | 0.0707 | 0.0688 | 1.0276 |
| 7 | 17608 | 8971.9257 | 0.0779 | 742.0000 | 698.8548 | 0.0827 | 0.0779 | 1.0617 |
| 8 | 19682 | 8969.4856 | 0.1006 | 1044.0000 | 902.5022 | 0.1164 | 0.1006 | 1.1568 |
| 9 | 21786 | 8972.5020 | 0.1820 | 1500.0000 | 1632.7580 | 0.1672 | 0.1820 | 0.9187 |

### Actual versus expected

**age**

| Spline_Feature | Support_Lower | Support_Upper | Lower_Inclusive | Upper_Inclusive | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|---|---|---|
| age | -inf | 18.0000 | false | true | 191 | 49.8864 | 11.0000 | 10.0193 | 1.0979 |
| age | 18.0000 | 34.0000 | false | true | 42595 | 19187.1580 | 1730.0000 | 1740.6840 | 0.9939 |
| age | 34.0000 | 44.0000 | false | true | 42488 | 21859.6967 | 1549.0000 | 1534.0665 | 1.0097 |
| age | 44.0000 | 55.0000 | false | true | 44804 | 24608.4405 | 1794.0000 | 1856.9859 | 0.9661 |
| age | 55.0000 | inf | false | false | 39426 | 24020.0726 | 1482.0000 | 1487.0929 | 0.9966 |

**vehicle_age**

| Spline_Feature | Support_Lower | Support_Upper | Lower_Inclusive | Upper_Inclusive | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|---|---|---|
| vehicle_age | -inf | 0.0000 | false | true | 14198 | 4133.5166 | 285.0000 | 300.7833 | 0.9475 |
| vehicle_age | 0.0000 | 2.0000 | false | true | 32796 | 15754.2959 | 1159.0000 | 1123.5812 | 1.0315 |
| vehicle_age | 2.0000 | 6.0000 | false | true | 41858 | 23153.3951 | 1780.0000 | 1777.2484 | 1.0015 |
| vehicle_age | 6.0000 | 11.0000 | false | true | 41337 | 23928.1031 | 1892.0000 | 1935.6188 | 0.9775 |
| vehicle_age | 11.0000 | inf | false | false | 39315 | 22755.9435 | 1450.0000 | 1491.6169 | 0.9721 |

**bonus**

| Spline_Feature | Support_Lower | Support_Upper | Lower_Inclusive | Upper_Inclusive | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|---|---|---|
| bonus | -inf | 50.0000 | false | true | 96090 | 56322.3928 | 2806.0000 | 3174.8391 | 0.8838 |
| bonus | 50.0000 | 64.0000 | false | true | 31177 | 15618.6506 | 1402.0000 | 1013.8414 | 1.3829 |
| bonus | 64.0000 | inf | false | false | 42237 | 17784.2108 | 2358.0000 | 2440.1681 | 0.9663 |

**region**

| region | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio | region_Level |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 17466 | 7566.3725 | 616.0000 | 653.4599 | 0.9427 | R11 |
| 1 | -0.2623 | 788 | 302.7738 | 20.0000 | 18.2895 | 1.0935 | R21 |
| 2 | 0.0137 | 1979 | 887.7460 | 75.0000 | 79.7984 | 0.9399 | R22 |
| 3 | -0.0540 | 2157 | 769.9523 | 40.0000 | 56.2788 | 0.7107 | R23 |
| 4 | -0.1683 | 40207 | 25735.7253 | 1598.0000 | 1629.9244 | 0.9804 | R24 |
| 5 | -0.1231 | 2708 | 1662.8860 | 110.0000 | 114.5276 | 0.9605 | R25 |
| 6 | -0.1858 | 2556 | 1232.4323 | 89.0000 | 84.3549 | 1.0551 | R26 |
| 7 | -0.0780 | 6778 | 2889.2489 | 254.0000 | 230.5116 | 1.1019 | R31 |
| 8 | -0.2442 | 3245 | 2020.8586 | 130.0000 | 111.7758 | 1.1630 | R41 |
| 9 | -0.0536 | 571 | 312.4010 | 24.0000 | 23.9148 | 1.0036 | R42 |
| 10 | -0.2082 | 332 | 134.6810 | 7.0000 | 9.2735 | 0.7548 | R43 |
| 11 | -0.0602 | 9791 | 5518.0675 | 381.0000 | 398.3623 | 0.9564 | R52 |
| 12 | -0.1041 | 10556 | 6932.1967 | 467.0000 | 469.0886 | 0.9955 | R53 |
| 13 | -0.1098 | 4752 | 2748.1134 | 215.0000 | 196.0312 | 1.0968 | R54 |
| 14 | -0.0215 | 7767 | 3570.3315 | 236.0000 | 273.9805 | 0.8614 | R72 |
| 15 | -0.4303 | 4383 | 1820.8117 | 104.0000 | 90.8433 | 1.1448 | R73 |
| 16 | 0.1287 | 1131 | 609.9538 | 44.0000 | 50.9695 | 0.8633 | R74 |
| 17 | 0.1626 | 21080 | 11281.2383 | 1030.0000 | 1066.0825 | 0.9662 | R82 |
| 18 | -0.2034 | 1304 | 583.9605 | 30.0000 | 37.5664 | 0.7986 | R83 |
| 19 | -0.1087 | 8956 | 3723.0506 | 260.0000 | 266.6794 | 0.9750 | R91 |
| 20 | 0.0375 | 19846 | 8968.5326 | 801.0000 | 733.8302 | 1.0915 | R93 |
| 21 | -0.1442 | 1151 | 453.9201 | 35.0000 | 33.3054 | 1.0509 | R94 |

**fuel**

| fuel | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio | fuel_Level |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 83146 | 42910.0428 | 3359.0000 | 3388.5885 | 0.9913 | 'Diesel' |
| 1 | -0.1536 | 86358 | 46815.2113 | 3207.0000 | 3240.2601 | 0.9897 | 'Regular' |

## Rating tables

**intercept**

| Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|
| -2.7821 | -2.7821 | 0.0568 | estimated | 0.0619 |

**age**

| age | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 18.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 34.0000 | -0.3458 | -0.3458 | 0.0479 | estimated | 0.7077 |
| 44.0000 | 0.1123 | 0.1123 | 0.0424 | estimated | 1.1188 |
| 55.0000 | 0.0438 | 0.0438 | 0.0439 | estimated | 1.0448 |
| 100.0000 | 0.4541 | 0.4541 | 0.1068 | estimated | 1.5748 |

**vehicle_age**

| vehicle_age | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 0.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 2.0000 | 0.0282 | 0.0282 | 0.0287 | estimated | 1.0286 |
| 6.0000 | 0.1618 | 0.1618 | 0.0299 | estimated | 1.1757 |
| 11.0000 | 0.0528 | 0.0528 | 0.0281 | estimated | 1.0542 |
| 100.0000 | -1.8481 | -1.8481 | 1.2599 | estimated | 0.1575 |

**bonus**

| bonus | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 50.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 64.0000 | 0.4755 | 0.4755 | 0.0112 | estimated | 1.6088 |
| 230.0000 | 2.9070 | 2.9070 | 0.2266 | estimated | 18.3011 |

**region**

| region | Rating_Factor | region_Level | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | R11 | 0.0000 | 0.0000 | reference | 1.0000 |
| 1 | -0.2623 | R21 | -0.2623 | 0.1344 | estimated | 0.7693 |
| 2 | 0.0137 | R22 | 0.0137 | 0.0686 | estimated | 1.0138 |
| 3 | -0.0540 | R23 | -0.0540 | 0.0780 | estimated | 0.9474 |
| 4 | -0.1683 | R24 | -0.1683 | 0.0277 | estimated | 0.8451 |
| 5 | -0.1231 | R25 | -0.1231 | 0.0589 | estimated | 0.8842 |
| 6 | -0.1858 | R26 | -0.1858 | 0.0666 | estimated | 0.8304 |
| 7 | -0.0780 | R31 | -0.0780 | 0.0444 | estimated | 0.9250 |
| 8 | -0.2442 | R41 | -0.2442 | 0.0592 | estimated | 0.7833 |
| 9 | -0.0536 | R42 | -0.0536 | 0.1234 | estimated | 0.9478 |
| 10 | -0.2082 | R43 | -0.2082 | 0.1811 | estimated | 0.8121 |
| 11 | -0.0602 | R52 | -0.0602 | 0.0372 | estimated | 0.9416 |
| 12 | -0.1041 | R53 | -0.1041 | 0.0355 | estimated | 0.9011 |
| 13 | -0.1098 | R54 | -0.1098 | 0.0475 | estimated | 0.8960 |
| 14 | -0.0215 | R72 | -0.0215 | 0.0418 | estimated | 0.9787 |
| 15 | -0.4303 | R73 | -0.4303 | 0.0655 | estimated | 0.6503 |
| 16 | 0.1287 | R74 | 0.1287 | 0.0841 | estimated | 1.1373 |
| 17 | 0.1626 | R82 | 0.1626 | 0.0290 | estimated | 1.1766 |
| 18 | -0.2034 | R83 | -0.2034 | 0.0976 | estimated | 0.8160 |
| 19 | -0.1087 | R91 | -0.1087 | 0.0419 | estimated | 0.8970 |
| 20 | 0.0375 | R93 | 0.0375 | 0.0312 | estimated | 1.0382 |
| 21 | -0.1442 | R94 | -0.1442 | 0.1040 | estimated | 0.8657 |

**fuel**

| fuel | Rating_Factor | fuel_Level | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | 'Diesel' | 0.0000 | 0.0000 | reference | 1.0000 |
| 1 | -0.1536 | 'Regular' | -0.1536 | 0.0145 | estimated | 0.8576 |

## Plan

The model's source code. Save it, edit it, re-run it.

```json
{
  "family": "poisson",
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
