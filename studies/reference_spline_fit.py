"""Independent dense natural-cubic GLM references; requires NumPy and SciPy."""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import CubicSpline
from scipy.optimize import root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('tests/fixtures/spline_fit.json'))
    args = parser.parse_args()
    knots = np.array([0., .4, 1.7, 3.])
    x = np.linspace(-.7, 3.7, 160) + .003 * np.sin(np.arange(160))
    basis = CubicSpline(knots, np.eye(len(knots)), bc_type='natural', extrapolate=False)
    b = basis(np.clip(x, knots[0], knots[-1]))
    left, right = x < knots[0], x > knots[-1]
    b[left] = np.eye(4)[0] + (x[left] - knots[0])[:, None] * basis(knots[0], 1)
    b[right] = np.eye(4)[-1] + (x[right] - knots[-1])[:, None] * basis(knots[-1], 1)
    category = ((np.arange(len(x)) * 7) % 13 < 6).astype(float)
    design = np.column_stack([np.ones(len(x)), category, b[:, 1:]])
    weights = .2 + (np.arange(len(x)) % 11) / 5
    weights[::19] = 0.
    offset = .15 * np.sin(np.arange(len(x)) / 9)
    cases = []
    for family, power in [('gaussian', None), ('poisson', 1.), ('gamma', 2.), ('tweedie', 1.5), ('binomial', None)]:
        truth = np.array([.3, -.25, .2, -.3, .15])
        eta = design @ truth + offset
        if family == 'gaussian':
            target = eta + .12*np.sin(np.arange(len(x))*.9)
        elif family == 'binomial':
            target = 1/(1+np.exp(-eta)) + .035*np.sin(np.arange(len(x))*.9)
        else:
            target = np.exp(eta) * np.exp(.3*np.sin(np.arange(len(x))*.9))

        def mean(beta):
            e = design @ beta + offset
            return e if family == 'gaussian' else 1/(1+np.exp(-e)) if family == 'binomial' else np.exp(e)

        def score(beta):
            mu = mean(beta)
            residual = target-mu
            if power is not None:
                residual = residual * mu**(1-power)
            return design.T @ (weights*residual) / weights.sum()

        fit = root(score, truth, tol=1e-11)
        assert fit.success, fit.message
        assert np.max(np.abs(score(fit.x))) < 1e-12
        cases.append({'family': family, 'power': power, 'target': target.tolist(),
                      'coefficients': fit.x.tolist(), 'means': mean(fit.x).tolist(),
                      'maximum_weight_normalized_score': float(np.max(np.abs(score(fit.x))))})
    artifact = {'producer': 'SciPy CubicSpline plus independent dense GLM score root',
                'scipy_version': scipy.__version__, 'knots': knots.tolist(), 'x': x.tolist(),
                'category': category.astype(int).tolist(), 'weights': weights.tolist(),
                'offset': offset.tolist(), 'cases': cases}
    args.output.write_text(json.dumps(artifact, indent=2) + '\n')


if __name__ == '__main__':
    main()
