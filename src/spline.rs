//! Natural cubic geometry for exact continuous rating effects.
//!
//! Parameters are function values at finite ordered knots. Between knots the
//! function is cubic and C2; outside them it follows the boundary tangent. Local
//! coefficients use u=(x-left)/(right-left), avoiding powers of large raw inputs.
//! Curvature penalties are defined on the normalized whole knot range [0,1].
//!
//! This module is numerical groundwork, not yet a public Plan/scoring feature.

type SplineResult<T> = Result<T, &'static str>;

#[derive(Debug, Clone)]
struct Geometry {
    knots: Vec<f64>,
    widths: Vec<f64>,
    normalized_widths: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
enum Location {
    Knot(usize),
    Interior(usize, f64),
    Left(f64),
    Right(f64),
}

impl Geometry {
    fn new(knots: Vec<f64>) -> SplineResult<Self> {
        if knots.len() < 2
            || knots.iter().any(|v| !v.is_finite())
            || knots.windows(2).any(|v| v[0] >= v[1])
        {
            return Err("A natural spline needs at least two finite, strictly increasing knots");
        }
        let span = knots[knots.len() - 1] - knots[0];
        if !span.is_finite() || span <= 0.0 {
            return Err("Spline knot range is not representable in finite arithmetic");
        }
        let widths: Vec<f64> = knots.windows(2).map(|v| v[1] - v[0]).collect();
        let normalized_widths: Vec<f64> = widths.iter().map(|h| h / span).collect();
        if normalized_widths
            .iter()
            .any(|h| !h.is_finite() || *h <= 0.0)
        {
            return Err("Spline knot spacing cannot be represented on the normalized range");
        }
        Ok(Self {
            knots,
            widths,
            normalized_widths,
        })
    }

    fn locate(&self, x: f64) -> SplineResult<Location> {
        if !x.is_finite() {
            return Err("Spline predictors must be finite");
        }
        let last = self.knots.len() - 1;
        let result = match self.knots.binary_search_by(|k| k.partial_cmp(&x).unwrap()) {
            Ok(index) => Location::Knot(index),
            Err(0) => Location::Left((x - self.knots[0]) / self.widths[0]),
            Err(index) if index == self.knots.len() => {
                Location::Right((x - self.knots[last]) / self.widths[last - 1])
            }
            Err(index) => Location::Interior(
                index - 1,
                (x - self.knots[index - 1]) / self.widths[index - 1],
            ),
        };
        let position = match result {
            Location::Knot(_) => 0.0,
            Location::Interior(_, u) | Location::Left(u) | Location::Right(u) => u,
        };
        if !position.is_finite() {
            return Err("Spline extrapolation distance is nonfinite");
        }
        Ok(result)
    }

    /// Solve the tridiagonal natural-boundary system for second derivatives on
    /// the unit-range coordinate, then express each cubic on its local unit interval.
    fn coefficients(&self, values: &[f64]) -> SplineResult<Vec<[f64; 4]>> {
        let n = self.knots.len();
        if values.len() != n || values.iter().any(|v| !v.is_finite()) {
            return Err("Spline values must be finite and match the knot count");
        }
        let h = &self.normalized_widths;
        let mut diagonal = vec![0.0; n - 2];
        let mut rhs = vec![0.0; n - 2];
        for j in 0..n - 2 {
            let i = j + 1;
            diagonal[j] = 2.0 * (h[i - 1] + h[i]);
            rhs[j] =
                6.0 * ((values[i + 1] - values[i]) / h[i] - (values[i] - values[i - 1]) / h[i - 1]);
            if j > 0 {
                let factor = h[i - 1] / diagonal[j - 1];
                diagonal[j] -= factor * h[i - 1];
                rhs[j] -= factor * rhs[j - 1];
            }
            if !diagonal[j].is_finite() || diagonal[j] <= 0.0 || !rhs[j].is_finite() {
                return Err("Spline geometry/values produce an unusable curvature system");
            }
        }
        let mut second = vec![0.0; n];
        for j in (0..n - 2).rev() {
            second[j + 1] = (rhs[j] - h[j + 1] * second[j + 2]) / diagonal[j];
        }
        let result: Vec<[f64; 4]> = (0..n - 1)
            .map(|i| {
                let h2 = h[i] * h[i];
                [
                    values[i],
                    values[i + 1] - values[i] - h2 * (2.0 * second[i] + second[i + 1]) / 6.0,
                    h2 * second[i] / 2.0,
                    h2 * (second[i + 1] - second[i]) / 6.0,
                ]
            })
            .collect();
        if result.iter().flatten().any(|v| !v.is_finite()) {
            return Err("Spline polynomial coefficients are nonfinite");
        }
        Ok(result)
    }
}

fn polynomial(c: &[f64; 4], u: f64) -> f64 {
    ((c[3] * u + c[2]) * u + c[1]) * u + c[0]
}

/// Cardinal basis: basis j equals one at knot j and zero at the other knots.
/// Each column retains four local coefficients per interval. Fitting can aggregate
/// four local powers by interval, then transform those small sufficient statistics,
/// without storing an observations-by-knots design matrix.
#[derive(Debug, Clone)]
pub(crate) struct NaturalCubicBasis {
    geometry: Geometry,
    columns: Vec<Vec<[f64; 4]>>,
}

impl NaturalCubicBasis {
    pub(crate) fn new(knots: Vec<f64>) -> SplineResult<Self> {
        let geometry = Geometry::new(knots)?;
        let n = geometry.knots.len();
        let mut columns = Vec::with_capacity(n);
        for j in 0..n {
            let mut values = vec![0.0; n];
            values[j] = 1.0;
            columns.push(geometry.coefficients(&values)?);
        }
        Ok(Self { geometry, columns })
    }

    pub(crate) fn curve(&self, values: &[f64]) -> SplineResult<NaturalCubicCurve> {
        Ok(NaturalCubicCurve {
            geometry: self.geometry.clone(),
            values: values.to_vec(),
            coefficients: self.geometry.coefficients(values)?,
        })
    }

    pub(crate) fn weights(&self, x: f64) -> SplineResult<Vec<f64>> {
        let location = self.geometry.locate(x)?;
        let n = self.geometry.knots.len();
        let weights: Vec<f64> = self
            .columns
            .iter()
            .enumerate()
            .map(|(j, column)| match location {
                Location::Knot(index) => {
                    if index == j {
                        1.0
                    } else {
                        0.0
                    }
                }
                Location::Interior(i, u) => polynomial(&column[i], u),
                Location::Left(u) => (if j == 0 { 1.0 } else { 0.0 }) + column[0][1] * u,
                Location::Right(u) => {
                    let c = column[n - 2];
                    (if j == n - 1 { 1.0 } else { 0.0 }) + (c[1] + 2.0 * c[2] + 3.0 * c[3]) * u
                }
            })
            .collect();
        if weights.iter().any(|v| !v.is_finite()) {
            return Err("Spline basis evaluation is nonfinite");
        }
        Ok(weights)
    }

    /// R with b'Rb = integral_0^1 (d2 f / dt2)^2 dt, where t normalizes the
    /// entire knot range. This penalty is invariant to affine rescaling of x.
    /// Constant and linear functions are in its null space.
    pub(crate) fn curvature_penalty(&self) -> SplineResult<Vec<f64>> {
        let n = self.columns.len();
        let mut penalty = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let value: f64 = self
                    .geometry
                    .normalized_widths
                    .iter()
                    .enumerate()
                    .map(|(s, h)| {
                        let a = self.columns[i][s];
                        let b = self.columns[j][s];
                        (4.0 * a[2] * b[2] + 6.0 * (a[2] * b[3] + a[3] * b[2]) + 12.0 * a[3] * b[3])
                            / h
                            / h
                            / h
                    })
                    .sum();
                if !value.is_finite() {
                    return Err("Spline curvature penalty is nonfinite");
                }
                penalty[i * n + j] = value;
                penalty[j * n + i] = value;
            }
        }
        Ok(penalty)
    }

    /// Accumulate an IRLS block in interval-local coordinates, then transform to
    /// knot-value coefficients. `curvature` is the nonnegative working weight;
    /// `scores` are negative loss derivatives with respect to the linear predictor.
    /// Uses O(knots^2) working memory, independent of observation count.
    pub(crate) fn normal_equations(
        &self,
        predictors: &[f64],
        curvature: &[f64],
        scores: &[f64],
    ) -> SplineResult<(Vec<f64>, Vec<f64>)> {
        self.accumulate::<true>(predictors, curvature, scores)
    }

    /// Apply the transposed basis without constructing an information matrix.
    /// Normalization and convergence need only these knot-value loadings.
    pub(crate) fn loadings(
        &self,
        predictors: &[f64],
        scores: &[f64],
    ) -> SplineResult<Vec<f64>> {
        self.accumulate::<false>(predictors, &[], scores)
            .map(|(_, score)| score)
    }

    fn accumulate<const INFORMATION: bool>(
        &self,
        predictors: &[f64],
        curvature: &[f64],
        scores: &[f64],
    ) -> SplineResult<(Vec<f64>, Vec<f64>)> {
        if (INFORMATION && predictors.len() != curvature.len()) || predictors.len() != scores.len() {
            return Err("Spline predictors, curvatures and scores must have matching lengths");
        }
        let n = self.columns.len();
        // n-1 cubic intervals plus two linear tails. Exact-knot observations are
        // accumulated separately so cardinal knot weights remain exactly one/zero.
        let mut gram = vec![[0.0; 16]; if INFORMATION { n + 1 } else { 0 }];
        let mut gradient = vec![[0.0; 4]; n + 1];
        let mut knot_curvature = vec![0.0; if INFORMATION { n } else { 0 }];
        let mut knot_scores = vec![0.0; n];
        for (row, (&x, &g)) in predictors.iter().zip(scores).enumerate() {
            let w = if INFORMATION { curvature[row] } else { 0.0 };
            if !w.is_finite() || w < 0.0 || !g.is_finite() {
                return Err(
                    "Spline working curvatures must be finite/nonnegative and scores finite",
                );
            }
            let location = self.geometry.locate(x)?;
            let (slot, phi) = match location {
                Location::Knot(k) => {
                    if INFORMATION {
                        knot_curvature[k] += w;
                    }
                    knot_scores[k] += g;
                    continue;
                }
                Location::Interior(i, u) => (i, [1.0, u, u * u, u * u * u]),
                Location::Left(u) => (n - 1, [1.0, u, 0.0, 0.0]),
                Location::Right(u) => (n, [1.0, u, 0.0, 0.0]),
            };
            let root = w.sqrt();
            for i in 0..4 {
                gradient[slot][i] += g * phi[i];
                if INFORMATION {
                    for j in 0..4 {
                        gram[slot][4 * i + j] += (root * phi[i]) * (root * phi[j]);
                    }
                }
            }
        }
        let mut information = vec![0.0; if INFORMATION { n * n } else { 0 }];
        let mut score = knot_scores;
        if INFORMATION {
            for i in 0..n {
                information[i * n + i] = knot_curvature[i];
            }
        }
        for slot in 0..n + 1 {
            let mapping: Vec<[f64; 4]> = (0..n)
                .map(|j| {
                    if slot < n - 1 {
                        self.columns[j][slot]
                    } else if slot == n - 1 {
                        [
                            if j == 0 { 1.0 } else { 0.0 },
                            self.columns[j][0][1],
                            0.0,
                            0.0,
                        ]
                    } else {
                        let c = self.columns[j][n - 2];
                        [
                            if j == n - 1 { 1.0 } else { 0.0 },
                            c[1] + 2.0 * c[2] + 3.0 * c[3],
                            0.0,
                            0.0,
                        ]
                    }
                })
                .collect();
            for i in 0..n {
                for a in 0..4 {
                    score[i] += mapping[i][a] * gradient[slot][a];
                }
                if !INFORMATION {
                    continue;
                }
                for j in 0..=i {
                    let mut value = 0.0;
                    for a in 0..4 {
                        for b in 0..4 {
                            value += mapping[i][a] * gram[slot][a * 4 + b] * mapping[j][b];
                        }
                    }
                    information[i * n + j] += value;
                    if i != j {
                        information[j * n + i] += value;
                    }
                }
            }
        }
        if information.iter().chain(&score).any(|v| !v.is_finite()) {
            return Err("Spline normal equations are nonfinite");
        }
        Ok((information, score))
    }
}

/// One fitted curve, scoring in O(log knots) and retaining a reviewable local
/// polynomial representation. It has no family, fit status or inference by itself.
#[derive(Debug, Clone)]
pub(crate) struct NaturalCubicCurve {
    geometry: Geometry,
    values: Vec<f64>,
    coefficients: Vec<[f64; 4]>,
}

impl NaturalCubicCurve {
    /// Compile editable knot values directly, without constructing a cardinal basis.
    pub(crate) fn new(knots: Vec<f64>, values: Vec<f64>) -> SplineResult<Self> {
        let geometry = Geometry::new(knots)?;
        let coefficients = geometry.coefficients(&values)?;
        Ok(Self {
            geometry,
            values,
            coefficients,
        })
    }

    pub(crate) fn evaluate(&self, x: f64) -> SplineResult<f64> {
        self.derivative(x, 0)
    }

    pub(crate) fn derivative(&self, x: f64, order: usize) -> SplineResult<f64> {
        if order > 2 {
            return Err("Spline derivative order must be 0, 1 or 2");
        }
        let last = self.values.len() - 1;
        let location = self.geometry.locate(x)?;
        let local = |i: usize, u: f64| {
            let c = self.coefficients[i];
            let h = self.geometry.widths[i];
            match order {
                0 => polynomial(&c, u),
                1 => ((3.0 * c[3] * u + 2.0 * c[2]) * u + c[1]) / h,
                _ => (6.0 * c[3] * u + 2.0 * c[2]) / h / h,
            }
        };
        let result = match location {
            Location::Knot(i) if order == 0 => self.values[i],
            Location::Knot(0) => local(0, 0.0),
            Location::Knot(i) => local(i - 1, 1.0),
            Location::Interior(i, u) => local(i, u),
            Location::Left(u) => match order {
                0 => self.values[0] + self.coefficients[0][1] * u,
                1 => self.coefficients[0][1] / self.geometry.widths[0],
                _ => 0.0,
            },
            Location::Right(u) => {
                let c = self.coefficients[last - 1];
                let slope = c[1] + 2.0 * c[2] + 3.0 * c[3];
                match order {
                    0 => self.values[last] + slope * u,
                    1 => slope / self.geometry.widths[last - 1],
                    _ => 0.0,
                }
            }
        };
        if !result.is_finite() {
            return Err("Spline evaluation is nonfinite");
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-9 + 1e-10 * expected.abs(),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn agrees_with_independent_scipy_curves_and_integrated_curvature() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/natural_spline.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let numbers = |key: &str| {
                case[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect::<Vec<_>>()
            };
            let knots = numbers("knots");
            let values = numbers("values");
            let basis = NaturalCubicBasis::new(knots.clone()).unwrap();
            let curve = basis.curve(&values).unwrap();
            for row in case["probes"].as_array().unwrap() {
                let x = row["x"].as_f64().unwrap();
                for order in 0..=2 {
                    let actual = curve.derivative(x, order).unwrap();
                    let expected = row["derivatives"][order].as_f64().unwrap();
                    let atol = case["derivative_atol"][order].as_f64().unwrap();
                    assert!(
                        (actual - expected).abs() <= atol + 1e-10 * expected.abs(),
                        "case {} x={x} derivative {order}: {actual} != {expected}, atol={atol}",
                        case["case"]
                    );
                }
                let weights = basis.weights(x).unwrap();
                for (weight, expected) in weights.iter().zip(row["basis"].as_array().unwrap()) {
                    close(*weight, expected.as_f64().unwrap());
                }
                close(weights.iter().sum(), 1.0);
                close(
                    weights.iter().zip(&values).map(|(a, b)| a * b).sum(),
                    curve.evaluate(x).unwrap(),
                );
                close(weights.iter().zip(&knots).map(|(a, b)| a * b).sum(), x);
            }
            for (got, expected) in basis
                .curvature_penalty()
                .unwrap()
                .iter()
                .zip(numbers("curvature_penalty"))
            {
                close(*got, expected);
            }
            let points: Vec<f64> = case["probes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p["x"].as_f64().unwrap())
                .collect();
            let (information, score) = basis
                .normal_equations(
                    &points,
                    &numbers("working_curvature"),
                    &numbers("working_score"),
                )
                .unwrap();
            for (got, expected) in information.iter().zip(numbers("information")) {
                close(*got, expected);
            }
            for (got, expected) in score.iter().zip(numbers("score")) {
                close(*got, expected);
            }
            let loadings = basis.loadings(&points, &numbers("working_score")).unwrap();
            for (got, expected) in loadings.iter().zip(numbers("score")) {
                close(*got, expected);
            }
            assert_eq!(loadings, score);
            for (x, y) in knots.iter().zip(&values) {
                assert_eq!(curve.evaluate(*x).unwrap(), *y);
            }
        }
    }

    #[test]
    fn affine_curves_and_two_knot_geometry_have_zero_curvature() {
        for knots in [vec![-2.0, 3.0], vec![-3.0, -1.0, 0.25, 2.0, 4.0]] {
            let basis = NaturalCubicBasis::new(knots.clone()).unwrap();
            let values: Vec<f64> = knots.iter().map(|x| 1.0 + 2.0 * x).collect();
            let curve = basis.curve(&values).unwrap();
            for x in [-20.0, -2.0, 0.0, 1.3, 5.0, 30.0] {
                close(curve.evaluate(x).unwrap(), 1.0 + 2.0 * x);
                close(curve.derivative(x, 1).unwrap(), 2.0);
                close(curve.derivative(x, 2).unwrap(), 0.0);
            }
            let penalty = basis.curvature_penalty().unwrap();
            for i in 0..knots.len() {
                close(
                    (0..knots.len())
                        .map(|j| penalty[i * knots.len() + j] * values[j])
                        .sum(),
                    0.0,
                );
            }
        }
    }

    #[test]
    fn joins_are_c2_and_tail_curvature_is_zero() {
        let knots = vec![-2.0, -0.5, 0.0, 1.0, 3.0];
        let basis = NaturalCubicBasis::new(knots.clone()).unwrap();
        let curve = basis.curve(&[0.2, 0.8, -0.4, 0.6, 1.1]).unwrap();
        for i in 1..knots.len() - 1 {
            let a = curve.coefficients[i - 1];
            let b = curve.coefficients[i];
            let left = curve.geometry.widths[i - 1];
            let right = curve.geometry.widths[i];
            close(polynomial(&a, 1.0), b[0]);
            close((a[1] + 2.0 * a[2] + 3.0 * a[3]) / left, b[1] / right);
            close(
                (2.0 * a[2] + 6.0 * a[3]) / left / left,
                2.0 * b[2] / right / right,
            );
        }
        for x in [-20.0, -2.0, 3.0, 20.0] {
            close(curve.derivative(x, 2).unwrap(), 0.0);
        }
    }

    #[test]
    fn unit_range_penalty_and_basis_are_affine_invariant_and_edits_are_linear() {
        let knots = vec![-2.0, -0.5, 0.0, 1.0, 3.0];
        let transformed: Vec<f64> = knots.iter().map(|x| 1073741824.0 + 16.0 * x).collect();
        let original = NaturalCubicBasis::new(knots).unwrap();
        let scaled = NaturalCubicBasis::new(transformed).unwrap();
        let values = [0.2, 0.8, -0.4, 0.6, 1.1];
        let curve = original.curve(&values).unwrap();
        let transformed_curve = scaled.curve(&values).unwrap();
        assert_eq!(
            original.curvature_penalty().unwrap(),
            scaled.curvature_penalty().unwrap()
        );
        let mut revised = values;
        revised[2] += 0.7;
        let edited = original.curve(&revised).unwrap();
        for x in [-10.0, -1.5, -0.25, 0.125, 2.375, 8.0] {
            let tx = 1073741824.0 + 16.0 * x;
            assert_eq!(original.weights(x).unwrap(), scaled.weights(tx).unwrap());
            close(
                curve.evaluate(x).unwrap(),
                transformed_curve.evaluate(tx).unwrap(),
            );
            close(
                curve.derivative(x, 1).unwrap() / 16.0,
                transformed_curve.derivative(tx, 1).unwrap(),
            );
            close(
                curve.derivative(x, 2).unwrap() / 256.0,
                transformed_curve.derivative(tx, 2).unwrap(),
            );
            close(
                edited.evaluate(x).unwrap() - curve.evaluate(x).unwrap(),
                0.7 * original.weights(x).unwrap()[2],
            );
        }
    }

    #[test]
    fn malformed_and_nonfinite_inputs_fail_without_inventing_values() {
        for knots in [
            vec![],
            vec![0.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, f64::INFINITY],
            vec![f64::NAN, 1.0],
            vec![-f64::MAX, f64::MAX],
        ] {
            assert!(NaturalCubicBasis::new(knots).is_err());
        }
        let basis = NaturalCubicBasis::new(vec![0.0, 1.0, 2.0]).unwrap();
        assert!(basis.curve(&[0.0, 1.0]).is_err());
        assert!(basis.curve(&[0.0, 1.0, f64::NAN]).is_err());
        let curve = basis.curve(&[0.0, 1.0, 2.0]).unwrap();
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(basis.weights(x).is_err());
            assert!(curve.evaluate(x).is_err());
        }
        assert!(curve.derivative(0.0, 3).is_err());
        assert!(basis.normal_equations(&[0.0], &[], &[1.0]).is_err());
        assert!(basis.normal_equations(&[0.0], &[-1.0], &[1.0]).is_err());
        assert!(basis.normal_equations(&[0.0], &[1.0], &[f64::NAN]).is_err());
        assert!(basis.loadings(&[0.0], &[]).is_err());
        assert!(basis.loadings(&[0.0], &[f64::NAN]).is_err());
        assert!(basis.loadings(&[f64::INFINITY], &[1.0]).is_err());
    }
}
