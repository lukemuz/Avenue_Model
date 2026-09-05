"""Prespecified Poisson–Gamma grouped simulation: recovery, coverage and delivery."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from avenue_model import Candidate, CredibilityResult, Plan, compare_models, poisson_credibility


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20260905)
    n, strength = 6000, 2.
    truth = rng.gamma(strength, 1 / strength, n)
    support = 10 ** rng.uniform(-2, 2, n)
    labels = [f'territory_{i:05d}' for i in range(n)]
    data = pl.DataFrame({'territory': np.repeat(labels, 2),
                         'exposure': (support[:, None] * [.3, .7]).reshape(-1),
                         'claims': rng.poisson((truth * support)[:, None] * [.3, .7]).reshape(-1)})
    baseline = Plan.frequency('exposure').fit(pl.DataFrame({'exposure': [1., 1.], 'rate': [1., 1.]}), 'rate')
    result = poisson_credibility(data, baseline, group='territory', counts='claims', prior_strength=strength)
    posterior = result.posterior.sort('group')
    pooled = posterior['relativity'].to_numpy()
    unpooled = posterior['unpooled_relativity'].to_numpy()
    covered = (posterior['relativity_lower'].to_numpy() <= truth) & (truth <= posterior['relativity_upper'].to_numpy())
    checks = []
    for name, mask in [('all', np.ones(n, dtype=bool)), ('sparse', support < 1), ('supported', support >= 10)]:
        pooled_mse = float(np.mean((pooled[mask] - truth[mask]) ** 2))
        unpooled_mse = float(np.mean((unpooled[mask] - truth[mask]) ** 2))
        coverage = float(np.mean(covered[mask]))
        assert pooled_mse < unpooled_mse
        assert .92 < coverage < .98
        checks.append({'population': name, 'groups': int(mask.sum()), 'pooled_mse': pooled_mse,
                       'unpooled_mse': unpooled_mse, 'credible_interval_coverage': coverage})
    quotes = pl.DataFrame({'territory': labels, 'exposure': np.ones(n), 'future_claims': rng.poisson(truth)})
    result.save(output / 'credibility')
    loaded = CredibilityResult.load(output / 'credibility')
    np.testing.assert_allclose(loaded.model.predict(quotes).to_series(), pooled, rtol=1e-12)
    # A fixed prior update uses the ordinary Plan machinery and retains the group table.
    replay = Plan('poisson', exposure='exposure', exposure_role='offset').offset_model(result.model).fit(
        quotes.with_columns(pl.Series('posterior_mean', pooled)), 'posterior_mean')
    np.testing.assert_allclose(replay.predict(quotes).to_series(), pooled, rtol=1e-8)
    comparison = compare_models(quotes, {
        'baseline': Candidate(baseline, 'counts', True),
        'pooled': Candidate(loaded.model, 'counts', training_status='completed'),
        'unpooled': Candidate(unpooled, 'counts', training_status='completed')},
        target='future_claims', unit='counts', metric='squared_error')
    comparison.summary.write_csv(output / 'future_comparison.csv')
    record = {'status': 'passed', 'seed': 20260905, 'groups': n, 'prior_strength': strength,
              'checks': checks, 'limitations': ['well-specified prior-predictive simulation',
                  'known fixed baseline; prior strength prespecified', 'no hyperparameter or baseline estimation uncertainty',
                  'coverage is averaged over the generating prior, not a frequentist guarantee for every fixed group rate']}
    (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
