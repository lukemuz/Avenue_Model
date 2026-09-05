# Poisson model of `avenue_frequency`

**Usable with caveats** · plan `3fcd929a2088`

It explains 4.5% of the deviance on the data it was measured against, with actual over expected at 0.9902 and a Gini of 0.286. It is usable, with 3 caveats to carry forward — the first is: 2 of 10 equal-exposure buckets have actual over expected outside 10%: buckets 2, 4. These are descriptive flags: review exposure, claim support and loss volatility before attributing the pattern to a missing interaction or a mis-specified band.

## Findings

| Severity | Stage | Finding |
|---|---|---|
| medium | validation | 2 of 10 equal-exposure buckets have actual over expected outside 10%: buckets 2, 4. These are descriptive flags: review exposure, claim support and loss volatility before attributing the pattern to a missing interaction or a mis-specified band. |
| low | plan | 1 rows in table 'bonus' hold under 0.1% of exposure each, so their factors will be noisy. |
| low | validation | 1 table rows hold less than 0.10% of total exposure each. Their actual-versus-expected figures are noisy and should not be read as evidence on their own. |

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
| Converged | yes after 6 sweeps (score 1.73e-11) |
| Deviance | 123551.577154 against a null of 129197.537340 |
| Pseudo R-squared | 0.0437 |
| Parameters | 37 |
| Dispersion | 1.000000 |
| AIC | 152857.6508 |
| BIC | 153269.8026 |
| Table conditioning | 1.70 |

## Validation

| | |
|---|---|
| Rows | 169504 scored of 169504 |
| Actual / expected | 0.9902 |
| Gini | 0.2856 |
| Lift, top over bottom | 5.05x |
| Out-of-sample pseudo R-squared | 0.0446 |

### Calibration, by equal-exposure bucket

| bin | n | weight | mean_predicted | actual | expected | actual_rate | expected_rate | ae_ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 17316 | 10408.8873 | 0.0366 | 358.0000 | 380.6482 | 0.0344 | 0.0366 | 0.9405 |
| 1 | 12376 | 7612.7985 | 0.0453 | 346.0000 | 344.8614 | 0.0454 | 0.0453 | 1.0033 |
| 2 | 14622 | 9178.9932 | 0.0493 | 401.0000 | 452.6684 | 0.0437 | 0.0493 | 0.8859 |
| 3 | 17214 | 9685.6399 | 0.0528 | 518.0000 | 511.4818 | 0.0535 | 0.0528 | 1.0127 |
| 4 | 14508 | 8305.0565 | 0.0565 | 420.0000 | 468.8490 | 0.0506 | 0.0565 | 0.8958 |
| 5 | 17218 | 8985.7835 | 0.0601 | 560.0000 | 540.2590 | 0.0623 | 0.0601 | 1.0365 |
| 6 | 17126 | 9022.4065 | 0.0682 | 595.0000 | 615.0537 | 0.0659 | 0.0682 | 0.9674 |
| 7 | 18251 | 8583.2915 | 0.0856 | 760.0000 | 734.5951 | 0.0885 | 0.0856 | 1.0346 |
| 8 | 20238 | 8981.8885 | 0.1100 | 1051.0000 | 988.1577 | 0.1170 | 0.1100 | 1.0636 |
| 9 | 20635 | 8960.5088 | 0.1780 | 1557.0000 | 1594.5994 | 0.1738 | 0.1780 | 0.9764 |

### Actual versus expected

**age**

| age | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|
| 21.0000 | 0.0000 | 2788 | 1087.7915 | 217.0000 | 235.0993 | 0.9230 |
| 26.0000 | -0.5012 | 9458 | 4051.4911 | 474.0000 | 475.9131 | 0.9960 |
| 36.0000 | -0.6025 | 39123 | 18388.1585 | 1349.0000 | 1327.5194 | 1.0162 |
| 51.0000 | -0.1764 | 62625 | 33236.9776 | 2409.0000 | 2466.5718 | 0.9767 |
| 71.0000 | -0.1873 | 46844 | 26994.3335 | 1765.0000 | 1762.4218 | 1.0015 |
| inf | -0.1562 | 8666 | 5966.5019 | 352.0000 | 363.6483 | 0.9680 |

**vehicle_age**

| vehicle_age | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|
| 1.0000 | 0.0000 | 32038 | 12355.5095 | 854.0000 | 884.3422 | 0.9657 |
| 5.0000 | 0.0686 | 47933 | 25539.3130 | 1947.0000 | 1890.2798 | 1.0300 |
| 10.0000 | 0.1245 | 42826 | 24668.0687 | 1960.0000 | 1999.1232 | 0.9804 |
| 20.0000 | -0.0334 | 44568 | 25818.2153 | 1742.0000 | 1806.8910 | 0.9641 |
| inf | -0.5087 | 2139 | 1344.1477 | 63.0000 | 50.5374 | 1.2466 |

**bonus**

| bonus | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|
| 51.0000 | 0.0000 | 100120 | 58416.8518 | 2918.0000 | 3031.5022 | 0.9626 |
| 60.0000 | 0.5291 | 19508 | 9824.2876 | 782.0000 | 783.2488 | 0.9984 |
| 80.0000 | 0.8925 | 29813 | 13423.2612 | 1575.0000 | 1471.0077 | 1.0707 |
| 100.0000 | 1.1673 | 18166 | 7198.6213 | 990.0000 | 1044.6175 | 0.9477 |
| 150.0000 | 2.0113 | 1852 | 844.2141 | 293.0000 | 290.7700 | 1.0077 |
| inf | 2.4938 | 45 | 18.0182 | 8.0000 | 10.0275 | 0.7978 |

**region**

| region | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio | region_Level |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 17466 | 7566.3725 | 616.0000 | 657.0554 | 0.9375 | R11 |
| 1 | -0.2606 | 788 | 302.7738 | 20.0000 | 18.4521 | 1.0839 | R21 |
| 2 | 0.0130 | 1979 | 887.7460 | 75.0000 | 79.5473 | 0.9428 | R22 |
| 3 | -0.0776 | 2157 | 769.9523 | 40.0000 | 56.0512 | 0.7136 | R23 |
| 4 | -0.1681 | 40207 | 25735.7253 | 1598.0000 | 1625.8571 | 0.9829 | R24 |
| 5 | -0.1273 | 2708 | 1662.8860 | 110.0000 | 113.9787 | 0.9651 | R25 |
| 6 | -0.1884 | 2556 | 1232.4323 | 89.0000 | 83.8508 | 1.0614 | R26 |
| 7 | -0.0789 | 6778 | 2889.2489 | 254.0000 | 230.9001 | 1.1000 | R31 |
| 8 | -0.2277 | 3245 | 2020.8586 | 130.0000 | 110.7317 | 1.1740 | R41 |
| 9 | -0.0333 | 571 | 312.4010 | 24.0000 | 24.1769 | 0.9927 | R42 |
| 10 | -0.1864 | 332 | 134.6810 | 7.0000 | 9.5385 | 0.7339 | R43 |
| 11 | -0.0702 | 9791 | 5518.0675 | 381.0000 | 398.4555 | 0.9562 | R52 |
| 12 | -0.1042 | 10556 | 6932.1967 | 467.0000 | 470.0897 | 0.9934 | R53 |
| 13 | -0.1192 | 4752 | 2748.1134 | 215.0000 | 196.3145 | 1.0952 | R54 |
| 14 | -0.0346 | 7767 | 3570.3315 | 236.0000 | 271.8508 | 0.8681 | R72 |
| 15 | -0.4502 | 4383 | 1820.8117 | 104.0000 | 90.9000 | 1.1441 | R73 |
| 16 | 0.1231 | 1131 | 609.9538 | 44.0000 | 51.7244 | 0.8507 | R74 |
| 17 | 0.1473 | 21080 | 11281.2383 | 1030.0000 | 1064.5792 | 0.9675 | R82 |
| 18 | -0.2106 | 1304 | 583.9605 | 30.0000 | 37.4852 | 0.8003 | R83 |
| 19 | -0.1127 | 8956 | 3723.0506 | 260.0000 | 267.5474 | 0.9718 | R91 |
| 20 | 0.0265 | 19846 | 8968.5326 | 801.0000 | 738.4865 | 1.0847 | R93 |
| 21 | -0.1399 | 1151 | 453.9201 | 35.0000 | 33.6008 | 1.0416 | R94 |

**fuel**

| fuel | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio | fuel_Level |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 83146 | 42910.0428 | 3359.0000 | 3385.3550 | 0.9922 | 'Diesel' |
| 1 | -0.1519 | 86358 | 46815.2113 | 3207.0000 | 3245.8187 | 0.9880 | 'Regular' |

## Rating tables

**intercept**

| Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|
| -2.6347 | -2.6347 | 0.0534 | estimated | 0.0717 |

**age**

| age | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 21.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 26.0000 | -0.5012 | -0.5012 | 0.0464 | estimated | 0.6058 |
| 36.0000 | -0.6025 | -0.6025 | 0.0432 | estimated | 0.5474 |
| 51.0000 | -0.1764 | -0.1764 | 0.0429 | estimated | 0.8382 |
| 71.0000 | -0.1873 | -0.1873 | 0.0441 | estimated | 0.8292 |
| inf | -0.1562 | -0.1562 | 0.0518 | estimated | 0.8554 |

**vehicle_age**

| vehicle_age | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 5.0000 | 0.0686 | 0.0686 | 0.0237 | estimated | 1.0710 |
| 10.0000 | 0.1245 | 0.1245 | 0.0238 | estimated | 1.1326 |
| 20.0000 | -0.0334 | -0.0334 | 0.0245 | estimated | 0.9671 |
| inf | -0.5087 | -0.5087 | 0.0859 | estimated | 0.6013 |

**bonus**

| bonus | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 51.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 60.0000 | 0.5291 | 0.5291 | 0.0238 | estimated | 1.6974 |
| 80.0000 | 0.8925 | 0.8925 | 0.0201 | estimated | 2.4412 |
| 100.0000 | 1.1673 | 1.1673 | 0.0245 | estimated | 3.2132 |
| 150.0000 | 2.0113 | 2.0113 | 0.0371 | estimated | 7.4728 |
| inf | 2.4938 | 2.4938 | 0.1484 | estimated | 12.1075 |

**region**

| region | Rating_Factor | region_Level | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | R11 | 0.0000 | 0.0000 | reference | 1.0000 |
| 1 | -0.2606 | R21 | -0.2606 | 0.1344 | estimated | 0.7706 |
| 2 | 0.0130 | R22 | 0.0130 | 0.0686 | estimated | 1.0130 |
| 3 | -0.0776 | R23 | -0.0776 | 0.0780 | estimated | 0.9253 |
| 4 | -0.1681 | R24 | -0.1681 | 0.0276 | estimated | 0.8453 |
| 5 | -0.1273 | R25 | -0.1273 | 0.0588 | estimated | 0.8805 |
| 6 | -0.1884 | R26 | -0.1884 | 0.0666 | estimated | 0.8283 |
| 7 | -0.0789 | R31 | -0.0789 | 0.0444 | estimated | 0.9241 |
| 8 | -0.2277 | R41 | -0.2277 | 0.0592 | estimated | 0.7964 |
| 9 | -0.0333 | R42 | -0.0333 | 0.1234 | estimated | 0.9673 |
| 10 | -0.1864 | R43 | -0.1864 | 0.1811 | estimated | 0.8299 |
| 11 | -0.0702 | R52 | -0.0702 | 0.0371 | estimated | 0.9323 |
| 12 | -0.1042 | R53 | -0.1042 | 0.0354 | estimated | 0.9011 |
| 13 | -0.1192 | R54 | -0.1192 | 0.0475 | estimated | 0.8876 |
| 14 | -0.0346 | R72 | -0.0346 | 0.0418 | estimated | 0.9660 |
| 15 | -0.4502 | R73 | -0.4502 | 0.0655 | estimated | 0.6375 |
| 16 | 0.1231 | R74 | 0.1231 | 0.0841 | estimated | 1.1310 |
| 17 | 0.1473 | R82 | 0.1473 | 0.0290 | estimated | 1.1587 |
| 18 | -0.2106 | R83 | -0.2106 | 0.0976 | estimated | 0.8101 |
| 19 | -0.1127 | R91 | -0.1127 | 0.0419 | estimated | 0.8934 |
| 20 | 0.0265 | R93 | 0.0265 | 0.0312 | estimated | 1.0268 |
| 21 | -0.1399 | R94 | -0.1399 | 0.1040 | estimated | 0.8695 |

**fuel**

| fuel | Rating_Factor | fuel_Level | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | 'Diesel' | 0.0000 | 0.0000 | reference | 1.0000 |
| 1 | -0.1519 | 'Regular' | -0.1519 | 0.0145 | estimated | 0.8590 |

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
