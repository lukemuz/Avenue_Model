# Gamma model of `avenue_severity`

**Not usable as it stands** · plan `c60bf8a40366`

This model should not be used as it stands. Overall actual over expected is 1.4414 (19291839.8900 actual against 13384135.3648 expected). Investigate population differences, loss volatility and model specification before changing the rate level. It explains 12.1% of the deviance on the data it was measured against, with actual over expected at 1.4414 and a Gini of 0.276.

## Findings

| Severity | Stage | Finding |
|---|---|---|
| high | validation | Overall actual over expected is 1.4414 (19291839.8900 actual against 13384135.3648 expected). Investigate population differences, loss volatility and model specification before changing the rate level. |
| medium | validation | 7 of 10 equal-exposure buckets have actual over expected outside 10%: buckets 0, 1, 2, 3, 6, 8, 9. These are descriptive flags: review exposure, claim support and loss volatility before attributing the pattern to a missing interaction or a mis-specified band. |

## The model

| Term | Kind | Rows | Parameters | Base |
|---|---|---:|---:|---|
| intercept | intercept | 1 | 1 | — |
| age | banded | 6 | 5 | lowest band |
| vehicle_age | banded | 5 | 4 | lowest band |
| bonus | banded | 6 | 5 | lowest band |
| region | categorical | 22 | 21 | R11 |
| fuel | categorical | 2 | 1 | 'Diesel' |

## Fit

| | |
|---|---|
| Converged | yes after 14 sweeps (score 2.60e-10) |
| Deviance | 28070.755300 against a null of 29689.614243 |
| Pseudo R-squared | 0.0545 |
| Parameters | 37 |
| Dispersion | 17.267770 |
| AIC | 371936.5979 |
| BIC | 372226.6667 |
| Table conditioning | 1.74 |

## Validation

| | |
|---|---|
| Rows | 6180 scored of 6180 |
| Actual / expected | 1.4414 |
| Gini | 0.2762 |
| Lift, top over bottom | 5.46x |
| Out-of-sample pseudo R-squared | 0.1211 |

### Calibration, by equal-exposure bucket

| bin | n | weight | mean_predicted | actual | expected | actual_rate | expected_rate | ae_ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 625 | 658.0000 | 1419.3216 | 1342010.4100 | 933913.6247 | 2039.5295 | 1419.3216 | 1.4370 |
| 1 | 637 | 661.0000 | 1614.4857 | 1327031.8900 | 1067175.0606 | 2007.6125 | 1614.4857 | 1.2435 |
| 2 | 611 | 653.0000 | 1715.9174 | 1977104.0500 | 1120494.0340 | 3027.7244 | 1715.9174 | 1.7645 |
| 3 | 611 | 657.0000 | 1779.2953 | 1384087.8700 | 1168997.0236 | 2106.6786 | 1779.2953 | 1.1840 |
| 4 | 618 | 654.0000 | 1844.6602 | 1109963.6500 | 1206407.8012 | 1697.1921 | 1844.6602 | 0.9201 |
| 5 | 624 | 658.0000 | 1923.6389 | 1282316.2200 | 1265754.3758 | 1948.8088 | 1923.6389 | 1.0131 |
| 6 | 626 | 656.0000 | 2013.4319 | 906718.3100 | 1320811.3411 | 1382.1925 | 2013.4319 | 0.6865 |
| 7 | 622 | 664.0000 | 2124.0881 | 1470746.9400 | 1410394.4702 | 2214.9803 | 2124.0881 | 1.0428 |
| 8 | 597 | 649.0000 | 2322.1690 | 1183255.2300 | 1507087.6548 | 1823.1976 | 2322.1690 | 0.7851 |
| 9 | 609 | 656.0000 | 3632.7744 | 7308605.3200 | 2383099.9788 | 11141.1666 | 3632.7744 | 3.0668 |

### Actual versus expected

**age**

| age | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|
| 21.0000 | 0.0000 | 202 | 217.0000 | 6304629.7900 | 934139.5698 | 6.7491 |
| 26.0000 | -0.7687 | 437 | 474.0000 | 1016281.5800 | 945701.7394 | 1.0746 |
| 36.0000 | -0.7230 | 1278 | 1349.0000 | 2596216.7700 | 2739219.2975 | 0.9478 |
| 51.0000 | -0.8138 | 2278 | 2409.0000 | 5173884.9700 | 4322290.9416 | 1.1970 |
| 71.0000 | -0.7579 | 1657 | 1765.0000 | 3423337.7900 | 3369406.3824 | 1.0160 |
| inf | -0.2859 | 328 | 352.0000 | 777488.9900 | 1073377.4341 | 0.7243 |

**vehicle_age**

| vehicle_age | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|
| 1.0000 | 0.0000 | 793 | 854.0000 | 1547631.1300 | 1860401.8518 | 0.8319 |
| 5.0000 | -0.0102 | 1818 | 1947.0000 | 3477824.6300 | 4167454.4321 | 0.8345 |
| 10.0000 | -0.1640 | 1854 | 1960.0000 | 4808057.9900 | 3686828.9438 | 1.3041 |
| 20.0000 | -0.1217 | 1656 | 1742.0000 | 9389319.5600 | 3489658.1180 | 2.6906 |
| inf | 0.2198 | 59 | 63.0000 | 69006.5800 | 179792.0190 | 0.3838 |

**bonus**

| bonus | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|
| 51.0000 | 0.0000 | 2765 | 2918.0000 | 5720704.0600 | 5640707.6199 | 1.0142 |
| 60.0000 | 0.0503 | 746 | 782.0000 | 1340856.9700 | 1551302.8423 | 0.8643 |
| 80.0000 | 0.0157 | 1480 | 1575.0000 | 4011717.8400 | 3025355.2595 | 1.3260 |
| 100.0000 | 0.1793 | 924 | 990.0000 | 7569187.6800 | 2539308.9529 | 2.9808 |
| 150.0000 | -0.0250 | 259 | 293.0000 | 603354.6500 | 613669.4068 | 0.9832 |
| inf | -0.0492 | 6 | 8.0000 | 46018.6900 | 13791.2835 | 3.3368 |

**region**

| region | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio | region_Level |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 574 | 616.0000 | 896161.0100 | 1151514.9799 | 0.7782 | R11 |
| 1 | 1.4068 | 18 | 20.0000 | 32707.7500 | 152439.2912 | 0.2146 | R21 |
| 2 | -0.1161 | 74 | 75.0000 | 289324.4000 | 129291.2045 | 2.2378 | R22 |
| 3 | -0.1643 | 36 | 40.0000 | 57783.2300 | 65904.5112 | 0.8768 | R23 |
| 4 | 0.0977 | 1542 | 1598.0000 | 8778694.1200 | 3380466.3082 | 2.5969 | R24 |
| 5 | -0.0626 | 102 | 110.0000 | 354281.1500 | 190425.2362 | 1.8605 | R25 |
| 6 | -0.0236 | 81 | 89.0000 | 180691.3000 | 161114.9664 | 1.1215 | R26 |
| 7 | -0.0261 | 232 | 254.0000 | 427194.0800 | 477785.3172 | 0.8941 | R31 |
| 8 | -0.0069 | 124 | 130.0000 | 425235.8100 | 246252.4839 | 1.7268 | R41 |
| 9 | -0.3493 | 21 | 24.0000 | 34115.8500 | 34116.9208 | 1.0000 | R42 |
| 10 | 0.0582 | 7 | 7.0000 | 8823.9500 | 12713.6806 | 0.6941 | R43 |
| 11 | -0.1241 | 362 | 381.0000 | 637352.1500 | 622154.6615 | 1.0244 | R52 |
| 12 | 0.0995 | 451 | 467.0000 | 927553.8700 | 958078.1630 | 0.9681 | R53 |
| 13 | -0.0762 | 208 | 215.0000 | 313067.3300 | 399083.3128 | 0.7845 | R54 |
| 14 | -0.0211 | 217 | 236.0000 | 467487.4500 | 449655.2870 | 1.0397 | R72 |
| 15 | -0.1743 | 99 | 104.0000 | 128607.6300 | 160069.3156 | 0.8034 | R73 |
| 16 | -0.3290 | 41 | 44.0000 | 85518.3500 | 58084.6042 | 1.4723 | R74 |
| 17 | 0.1331 | 974 | 1030.0000 | 2790694.5100 | 2221370.3460 | 1.2563 | R82 |
| 18 | -0.3296 | 30 | 30.0000 | 36463.6700 | 39577.7425 | 0.9213 | R83 |
| 19 | 0.0612 | 232 | 260.0000 | 439727.3400 | 506416.8952 | 0.8683 | R91 |
| 20 | 0.1962 | 725 | 801.0000 | 1910384.5200 | 1830502.3582 | 1.0436 | R93 |
| 21 | 0.7588 | 30 | 35.0000 | 69970.4200 | 137117.7786 | 0.5103 | R94 |

**fuel**

| fuel | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio | fuel_Level |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 3162 | 3359.0000 | 7378460.2600 | 6715697.5155 | 1.0987 | 'Diesel' |
| 1 | -0.0059 | 3018 | 3207.0000 | 11913379.6300 | 6668437.8493 | 1.7865 | 'Regular' |

## Rating tables

**intercept**

| Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|
| 8.2911 | 8.2911 | 0.2196 | estimated | 3988.0585 |

**age**

| age | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 21.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 26.0000 | -0.7687 | -0.7687 | 0.1924 | estimated | 0.4636 |
| 36.0000 | -0.7230 | -0.7230 | 0.1802 | estimated | 0.4853 |
| 51.0000 | -0.8138 | -0.8138 | 0.1804 | estimated | 0.4432 |
| 71.0000 | -0.7579 | -0.7579 | 0.1847 | estimated | 0.4686 |
| inf | -0.2859 | -0.2859 | 0.2156 | estimated | 0.7513 |

**vehicle_age**

| vehicle_age | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 5.0000 | -0.0102 | -0.0102 | 0.0984 | estimated | 0.9899 |
| 10.0000 | -0.1640 | -0.1640 | 0.0985 | estimated | 0.8487 |
| 20.0000 | -0.1217 | -0.1217 | 0.1019 | estimated | 0.8854 |
| inf | 0.2198 | 0.2198 | 0.3570 | estimated | 1.2459 |

**bonus**

| bonus | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 51.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 60.0000 | 0.0503 | 0.0503 | 0.0978 | estimated | 1.0516 |
| 80.0000 | 0.0157 | 0.0157 | 0.0812 | estimated | 1.0158 |
| 100.0000 | 0.1793 | 0.1793 | 0.1039 | estimated | 1.1963 |
| 150.0000 | -0.0250 | -0.0250 | 0.1552 | estimated | 0.9753 |
| inf | -0.0492 | -0.0492 | 0.6173 | estimated | 0.9520 |

**region**

| region | Rating_Factor | region_Level | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | R11 | 0.0000 | 0.0000 | reference | 1.0000 |
| 1 | 1.4068 | R21 | 1.4068 | 0.5586 | estimated | 4.0829 |
| 2 | -0.1161 | R22 | -0.1161 | 0.2851 | estimated | 0.8904 |
| 3 | -0.1643 | R23 | -0.1643 | 0.3241 | estimated | 0.8485 |
| 4 | 0.0977 | R24 | 0.0977 | 0.1134 | estimated | 1.1026 |
| 5 | -0.0626 | R25 | -0.0626 | 0.2444 | estimated | 0.9393 |
| 6 | -0.0236 | R26 | -0.0236 | 0.2768 | estimated | 0.9766 |
| 7 | -0.0261 | R31 | -0.0261 | 0.1846 | estimated | 0.9742 |
| 8 | -0.0069 | R41 | -0.0069 | 0.2458 | estimated | 0.9931 |
| 9 | -0.3493 | R42 | -0.3493 | 0.5131 | estimated | 0.7052 |
| 10 | 0.0582 | R43 | 0.0582 | 0.7526 | estimated | 1.0599 |
| 11 | -0.1241 | R52 | -0.1241 | 0.1537 | estimated | 0.8833 |
| 12 | 0.0995 | R53 | 0.0995 | 0.1466 | estimated | 1.1046 |
| 13 | -0.0762 | R54 | -0.0762 | 0.1967 | estimated | 0.9267 |
| 14 | -0.0211 | R72 | -0.0211 | 0.1735 | estimated | 0.9791 |
| 15 | -0.1743 | R73 | -0.1743 | 0.2722 | estimated | 0.8400 |
| 16 | -0.3290 | R74 | -0.3290 | 0.3494 | estimated | 0.7196 |
| 17 | 0.1331 | R82 | 0.1331 | 0.1200 | estimated | 1.1424 |
| 18 | -0.3296 | R83 | -0.3296 | 0.4060 | estimated | 0.7192 |
| 19 | 0.0612 | R91 | 0.0612 | 0.1742 | estimated | 1.0631 |
| 20 | 0.1962 | R93 | 0.1962 | 0.1296 | estimated | 1.2168 |
| 21 | 0.7588 | R94 | 0.7588 | 0.4328 | estimated | 2.1356 |

**fuel**

| fuel | Rating_Factor | fuel_Level | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | 'Diesel' | 0.0000 | 0.0000 | reference | 1.0000 |
| 1 | -0.0059 | 'Regular' | -0.0059 | 0.0604 | estimated | 0.9942 |

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
      "kind": "banded",
      "column": "age",
      "breaks": {
        "kind": "explicit",
        "edges": [
          21.0,
          26.0,
          36.0,
          51.0,
          71.0
        ]
      }
    },
    {
      "kind": "banded",
      "column": "vehicle_age",
      "breaks": {
        "kind": "explicit",
        "edges": [
          1.0,
          5.0,
          10.0,
          20.0
        ]
      }
    },
    {
      "kind": "banded",
      "column": "bonus",
      "breaks": {
        "kind": "explicit",
        "edges": [
          51.0,
          60.0,
          80.0,
          100.0,
          150.0
        ]
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
