"""Regenerate the independent natural-cubic kernel fixture with SciPy.

SciPy's natural boundary condition is used inside the knot range. Outside it this
fixture explicitly extends endpoint tangents; it does not use cubic extrapolation.
Three-point Gaussian quadrature integrates every product of second derivatives
exactly within each interval (the product is a polynomial of degree two).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import CubicSpline


def fixture():
    geometries = [
        [-2., 3.],
        [-1., .25, 2.],
        [-3., -1.25, .1, .4, 2., 5.],
        [1e9+3.5*x for x in [-3., -1.25, .1, .4, 2., 5.]],
        [-2., -1.999, 0., .2, 3.],
    ]
    cases = []
    nodes, weights = np.polynomial.legendre.leggauss(3)
    for case, raw in enumerate(geometries):
        knots = np.array(raw)
        n = len(knots)
        values = np.sin(np.arange(n)*1.7)+.2*np.arange(n)
        curve = CubicSpline(knots, values, bc_type='natural', extrapolate=False)
        basis = CubicSpline(knots, np.eye(n), bc_type='natural', extrapolate=False)
        span = knots[-1]-knots[0]

        def evaluate(spline, x, order=0):
            if x < knots[0] or x > knots[-1]:
                edge = knots[0] if x < knots[0] else knots[-1]
                return (spline(edge)+spline(edge, 1)*(x-edge) if order == 0 else
                        spline(edge, 1) if order == 1 else np.zeros_like(spline(edge)))
            return spline(x, order)

        points = np.unique(np.concatenate([
            np.linspace(knots[0], knots[-1], 31),
            np.nextafter(knots, -np.inf), knots, np.nextafter(knots, np.inf),
            [knots[0]-2*span, knots[-1]+2*span],
        ]))
        probes = [{'x': float(x), 'derivatives': [float(evaluate(curve, x, order)) for order in range(3)],
                   'basis': evaluate(basis, x).tolist()} for x in points]
        design = np.array([row['basis'] for row in probes])
        working_curvature = .1+np.abs(np.sin(np.arange(len(probes))))
        working_score = np.cos(np.arange(len(probes)))
        penalty = np.zeros((n, n))
        for interval, (left, right) in enumerate(zip(knots[:-1], knots[1:])):
            h = right-left
            for node, weight in zip(nodes, weights):
                # Evaluate SciPy's local polynomial directly: adding a small
                # quadrature displacement to a 1e9 origin would round the node.
                local = (node+1)*h/2
                second = 6*basis.c[0, interval]*local+2*basis.c[1, interval]
                penalty += weight*h/2*np.outer(second, second)*span**3
        cases.append({'case': case, 'knots': knots.tolist(), 'values': values.tolist(),
                      # Differentiation amplifies interpolation roundoff by inverse
                      # knot spacing. In the clustered case SciPy's second derivative
                      # at the natural endpoint is ~3e-9 rather than exactly zero.
                      # Keep value tolerance fixed; bound derivative reference noise
                      # from input scale/spacing, independently of Avenue outputs.
                      'derivative_atol': [max(1e-9, 64*np.finfo(float).eps*max(1., np.max(np.abs(values)))/np.min(np.diff(knots))**order)
                                          for order in range(3)],
                      'working_curvature': working_curvature.tolist(), 'working_score': working_score.tolist(),
                      'information': (design.T@(working_curvature[:, None]*design)).ravel().tolist(),
                      'score': (design.T@working_score).tolist(),
                      'probes': probes, 'curvature_penalty': penalty.ravel().tolist()})
    return {'producer': 'studies/reference_natural_spline.py', 'scipy_version': scipy.__version__,
            'boundary_condition': 'natural', 'extrapolation': 'linear endpoint tangents',
            'penalty_coordinate': 'unit whole knot range', 'cases': cases}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(fixture(), allow_nan=False, indent=2)+'\n')
