# Tweedie model of `avenue_pure_premium`

**Not usable as it stands** · plan `fe6a8e7295d4`

This model should not be used as it stands. Overall actual over expected is 1.4188 (19291839.8900 actual against 13597670.9498 expected). Investigate population differences, loss volatility and model specification before changing the rate level. It explains 13.3% of the deviance on the data it was measured against, with actual over expected at 1.4188 and a Gini of 0.509.

## Findings

| Severity | Stage | Finding |
|---|---|---|
| high | validation | Overall actual over expected is 1.4188 (19291839.8900 actual against 13597670.9498 expected). Investigate population differences, loss volatility and model specification before changing the rate level. |
| medium | validation | 8 of 10 equal-exposure buckets have actual over expected outside 10%: buckets 0, 1, 3, 4, 5, 6, 7, 9. These are descriptive flags: review exposure, claim support and loss volatility before attributing the pattern to a missing interaction or a mis-specified band. |
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
| Converged | yes after 10 sweeps (score 9.36e-10) |
| Deviance | 20731794.676710 against a null of 22099342.683457 |
| Pseudo R-squared | 0.0619 |
| Parameters | 37 |
| Dispersion | 8664.809215 |
| Table conditioning | 1.70 |

## Validation

| | |
|---|---|
| Rows | 169504 scored of 169504 |
| Actual / expected | 1.4188 |
| Gini | 0.5090 |
| Lift, top over bottom | 27.14x |
| Out-of-sample pseudo R-squared | 0.1326 |

### Calibration, by equal-exposure bucket

| bin | n | weight | mean_predicted | actual | expected | actual_rate | expected_rate | ae_ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 14936 | 9072.4531 | 60.2376 | 349763.3800 | 546503.1792 | 38.5522 | 60.2376 | 0.6400 |
| 1 | 14496 | 9213.0313 | 75.0046 | 1113894.4900 | 691019.7108 | 120.9042 | 75.0046 | 1.6120 |
| 2 | 14599 | 8645.1115 | 85.4144 | 669575.4400 | 738416.7668 | 77.4513 | 85.4144 | 0.9068 |
| 3 | 15686 | 9013.4012 | 94.7975 | 1143536.3300 | 854447.8606 | 126.8707 | 94.7975 | 1.3383 |
| 4 | 15948 | 8942.7437 | 104.7570 | 718455.6900 | 936814.9866 | 80.3395 | 104.7570 | 0.7669 |
| 5 | 17452 | 9072.9482 | 118.8444 | 1207173.0900 | 1078268.8896 | 133.0519 | 118.8444 | 1.1195 |
| 6 | 16773 | 8862.1792 | 139.4693 | 1098970.7800 | 1236001.4915 | 124.0068 | 139.4693 | 0.8891 |
| 7 | 18447 | 9004.5589 | 167.2898 | 1815709.2800 | 1506370.9315 | 201.6433 | 167.2898 | 1.2054 |
| 8 | 19854 | 8952.7204 | 220.8389 | 1814990.7200 | 1977108.6436 | 202.7306 | 220.8389 | 0.9180 |
| 9 | 21313 | 8946.1065 | 450.7792 | 9359770.6900 | 4032718.4894 | 1046.2396 | 450.7792 | 2.3210 |

### Actual versus expected

**age**

| age | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|
| 21.0000 | 0.0000 | 2788 | 1087.7915 | 6304629.7900 | 1044457.2280 | 6.0363 |
| 26.0000 | -1.3219 | 9458 | 4051.4911 | 1016281.5800 | 920072.1972 | 1.1046 |
| 36.0000 | -1.3111 | 39123 | 18388.1585 | 2596216.7700 | 2764906.1157 | 0.9390 |
| 51.0000 | -0.9982 | 62625 | 33236.9776 | 5173884.9700 | 4464699.3177 | 1.1588 |
| 71.0000 | -0.9777 | 46844 | 26994.3335 | 3423337.7900 | 3317082.3916 | 1.0320 |
| inf | -0.4663 | 8666 | 5966.5019 | 777488.9900 | 1086453.6995 | 0.7156 |

**vehicle_age**

| vehicle_age | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|
| 1.0000 | 0.0000 | 32038 | 12355.5095 | 1547631.1300 | 2004577.1342 | 0.7720 |
| 5.0000 | 0.0529 | 47933 | 25539.3130 | 3477824.6300 | 4172445.2780 | 0.8335 |
| 10.0000 | -0.0614 | 42826 | 24668.0687 | 4808057.9900 | 3806518.4478 | 1.2631 |
| 20.0000 | -0.2033 | 44568 | 25818.2153 | 9389319.5600 | 3504798.2403 | 2.6790 |
| inf | -0.5391 | 2139 | 1344.1477 | 69006.5800 | 109331.8495 | 0.6312 |

**bonus**

| bonus | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio |
|---|---|---|---|---|---|---|
| 51.0000 | 0.0000 | 100120 | 58416.8518 | 5720704.0600 | 5823463.0033 | 0.9824 |
| 60.0000 | 0.5464 | 19508 | 9824.2876 | 1340856.9700 | 1522726.4304 | 0.8806 |
| 80.0000 | 0.8950 | 29813 | 13423.2612 | 4011717.8400 | 2817880.8253 | 1.4237 |
| 100.0000 | 1.3724 | 18166 | 7198.6213 | 7569187.6800 | 2792303.3216 | 2.7107 |
| 150.0000 | 1.9994 | 1852 | 844.2141 | 603354.6500 | 621202.6988 | 0.9713 |
| inf | 2.5347 | 45 | 18.0182 | 46018.6900 | 20094.6704 | 2.2901 |

**region**

| region | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio | region_Level |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 17466 | 7566.3725 | 896161.0100 | 1267957.7917 | 0.7068 | R11 |
| 1 | 1.4562 | 788 | 302.7738 | 32707.7500 | 198311.4273 | 0.1649 | R21 |
| 2 | -0.1237 | 1979 | 887.7460 | 289324.4000 | 135595.8221 | 2.1337 | R22 |
| 3 | -0.2012 | 2157 | 769.9523 | 57783.2300 | 94522.4457 | 0.6113 | R23 |
| 4 | -0.1060 | 40207 | 25735.7253 | 8778694.1200 | 3331562.6763 | 2.6350 | R24 |
| 5 | -0.1643 | 2708 | 1662.8860 | 354281.1500 | 211818.9180 | 1.6726 | R25 |
| 6 | -0.2378 | 2556 | 1232.4323 | 180691.3000 | 154815.6552 | 1.1671 | R26 |
| 7 | -0.0866 | 6778 | 2889.2489 | 427194.0800 | 449750.3559 | 0.9498 | R31 |
| 8 | -0.2784 | 3245 | 2020.8586 | 425235.8100 | 202193.6150 | 2.1031 | R41 |
| 9 | -0.5130 | 571 | 312.4010 | 34115.8500 | 30157.9270 | 1.1312 | R42 |
| 10 | -0.1170 | 332 | 134.6810 | 8823.9500 | 21352.7943 | 0.4132 | R43 |
| 11 | -0.2395 | 9791 | 5518.0675 | 637352.1500 | 637122.9612 | 1.0004 | R52 |
| 12 | -0.0373 | 10556 | 6932.1967 | 927553.8700 | 942010.2269 | 0.9847 | R53 |
| 13 | -0.2084 | 4752 | 2748.1134 | 313067.3300 | 353384.9055 | 0.8859 | R54 |
| 14 | -0.0655 | 7767 | 3570.3315 | 467487.4500 | 515416.2855 | 0.9070 | R72 |
| 15 | -0.6134 | 4383 | 1820.8117 | 128607.6300 | 147433.9900 | 0.8723 | R73 |
| 16 | -0.2334 | 1131 | 609.9538 | 85518.3500 | 65796.2593 | 1.2997 | R74 |
| 17 | 0.2664 | 21080 | 11281.2383 | 2790694.5100 | 2277374.3538 | 1.2254 | R82 |
| 18 | -0.5587 | 1304 | 583.9605 | 36463.6700 | 52036.8913 | 0.7007 | R83 |
| 19 | -0.0145 | 8956 | 3723.0506 | 439727.3400 | 565849.9941 | 0.7771 | R91 |
| 20 | 0.2587 | 19846 | 8968.5326 | 1910384.5200 | 1800690.8766 | 1.0609 | R93 |
| 21 | 0.6096 | 1151 | 453.9201 | 69970.4200 | 142514.7770 | 0.4910 | R94 |

**fuel**

| fuel | Rating_Factor | N | Exposure | Actual | Expected | AE_Ratio | fuel_Level |
|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 83146 | 42910.0428 | 7378460.2600 | 6911347.7553 | 1.0676 | 'Diesel' |
| 1 | -0.1891 | 86358 | 46815.2113 | 11913379.6300 | 6686323.1945 | 1.7818 | 'Regular' |

## Rating tables

**intercept**

| Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|
| 5.7215 | 5.7215 | 0.4154 | estimated | 305.3558 |

**age**

| age | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 21.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 26.0000 | -1.3219 | -1.3219 | 0.3740 | estimated | 0.2666 |
| 36.0000 | -1.3111 | -1.3111 | 0.3408 | estimated | 0.2695 |
| 51.0000 | -0.9982 | -0.9982 | 0.3425 | estimated | 0.3685 |
| 71.0000 | -0.9777 | -0.9777 | 0.3498 | estimated | 0.3762 |
| inf | -0.4663 | -0.4663 | 0.3876 | estimated | 0.6273 |

**vehicle_age**

| vehicle_age | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 1.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 5.0000 | 0.0529 | 0.0529 | 0.1703 | estimated | 1.0543 |
| 10.0000 | -0.0614 | -0.0614 | 0.1744 | estimated | 0.9404 |
| 20.0000 | -0.2033 | -0.2033 | 0.1782 | estimated | 0.8161 |
| inf | -0.5391 | -0.5391 | 0.5346 | estimated | 0.5833 |

**bonus**

| bonus | Rating_Factor | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|
| 51.0000 | 0.0000 | 0.0000 | 0.0000 | reference | 1.0000 |
| 60.0000 | 0.5464 | 0.5464 | 0.1768 | estimated | 1.7270 |
| 80.0000 | 0.8950 | 0.8950 | 0.1576 | estimated | 2.4474 |
| 100.0000 | 1.3724 | 1.3724 | 0.1962 | estimated | 3.9447 |
| 150.0000 | 1.9994 | 1.9994 | 0.3781 | estimated | 7.3847 |
| inf | 2.5347 | 2.5347 | 1.7618 | estimated | 12.6124 |

**region**

| region | Rating_Factor | region_Level | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | R11 | 0.0000 | 0.0000 | reference | 1.0000 |
| 1 | 1.4562 | R21 | 1.4562 | 0.6418 | estimated | 4.2897 |
| 2 | -0.1237 | R22 | -0.1237 | 0.5528 | estimated | 0.8836 |
| 3 | -0.2012 | R23 | -0.2012 | 0.6053 | estimated | 0.8177 |
| 4 | -0.1060 | R24 | -0.1060 | 0.2098 | estimated | 0.8994 |
| 5 | -0.1643 | R25 | -0.1643 | 0.4398 | estimated | 0.8485 |
| 6 | -0.2378 | R26 | -0.2378 | 0.4964 | estimated | 0.7884 |
| 7 | -0.0866 | R31 | -0.0866 | 0.3409 | estimated | 0.9170 |
| 8 | -0.2784 | R41 | -0.2784 | 0.4239 | estimated | 0.7570 |
| 9 | -0.5130 | R42 | -0.5130 | 1.0428 | estimated | 0.5987 |
| 10 | -0.1170 | R43 | -0.1170 | 1.3116 | estimated | 0.8896 |
| 11 | -0.2395 | R52 | -0.2395 | 0.2901 | estimated | 0.7870 |
| 12 | -0.0373 | R53 | -0.0373 | 0.2648 | estimated | 0.9634 |
| 13 | -0.2084 | R54 | -0.2084 | 0.3619 | estimated | 0.8119 |
| 14 | -0.0655 | R72 | -0.0655 | 0.3197 | estimated | 0.9366 |
| 15 | -0.6134 | R73 | -0.6134 | 0.4690 | estimated | 0.5415 |
| 16 | -0.2334 | R74 | -0.2334 | 0.7159 | estimated | 0.7919 |
| 17 | 0.2664 | R82 | 0.2664 | 0.2251 | estimated | 1.3053 |
| 18 | -0.5587 | R83 | -0.5587 | 0.7660 | estimated | 0.5720 |
| 19 | -0.0145 | R91 | -0.0145 | 0.3116 | estimated | 0.9856 |
| 20 | 0.2587 | R93 | 0.2587 | 0.2353 | estimated | 1.2952 |
| 21 | 0.6096 | R94 | 0.6096 | 0.6499 | estimated | 1.8396 |

**fuel**

| fuel | Rating_Factor | fuel_Level | Coefficient | Standard_Error | Status | Relativity |
|---|---|---|---|---|---|---|
| 0 | 0.0000 | 'Diesel' | 0.0000 | 0.0000 | reference | 1.0000 |
| 1 | -0.1891 | 'Regular' | -0.1891 | 0.1078 | estimated | 0.8277 |

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
