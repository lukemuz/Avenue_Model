"""Real-study booster arm, sharing the GLM study's train/holdout frames and units."""
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import numpy as np
import polars as pl
from avenue_model import Workbook, from_booster, resolve_lightgbm, tune_lgbm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_poisson_deviance


FEATURES = ['age', 'vehicle_age', 'bonus', 'region', 'fuel']


def run_challenger(train, holdout, baseline, directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    levels = {name: sorted(train[name].unique().to_list()) for name in ('region', 'fuel')}
    def codes(frame):
        return frame.select([pl.col(name).replace_strict({v: i for i, v in enumerate(levels[name])},
                                                       default=-1).cast(pl.Int32)
                             if name in levels else pl.col(name).cast(pl.Float64) for name in FEATURES])
    lgb, module_name = resolve_lightgbm()
    training_codes, holdout_codes = codes(train), codes(holdout)
    dataset = lgb.Dataset(training_codes.to_numpy(), label=train['avenue_frequency'].to_numpy(),
                          weight=train['exposure'].to_numpy(), feature_name=FEATURES,
                          categorical_feature=['region', 'fuel'])
    folds = list(GroupKFold(n_splits=3).split(train, groups=train['policy_id'].to_numpy()))
    started = time.perf_counter()
    tuning = tune_lgbm(dataset, {'objective': 'poisson', 'num_iterations': 80,
        'num_leaves': 4, 'max_depth': 2, 'min_data_in_leaf': 2000,
        'num_threads': 4, 'verbosity': -1, 'deterministic': True, 'force_col_wise': True,
        'seed': 20260905}, n_trials=4, seed=20260905,
        folds=folds,
        tunable=['learning_rate', 'interaction_penalty', 'interaction_complexity'],
        space={'learning_rate': (.05, .2), 'interaction_penalty': (0., 5.),
               'interaction_complexity': (0., 10.)})
    tuning_seconds = time.perf_counter() - started
    (out / 'trials.json').write_text(json.dumps([asdict(t) for t in tuning.trials], indent=2))
    (out / 'tuning.txt').write_text(tuning.summary())
    selected = tuning.best_cv
    started = time.perf_counter()
    booster = lgb.train({**selected.params, 'num_iterations': selected.num_iterations}, dataset)
    fit_seconds = time.perf_counter() - started
    booster.save_model(str(out / 'booster.txt'))
    # Probe every numerical split at its threshold and adjacent floating-point values.
    anchor = training_codes.row(0, named=True)
    probes = []
    def visit(node):
        if 'split_feature' not in node:
            return
        feature = FEATURES[node['split_feature']]
        if node['decision_type'] == '<=':
            threshold = float(node['threshold'])
            for value in (np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf), None):
                probes.append({**anchor, feature: value})
        visit(node['left_child'])
        visit(node['right_child'])
    for tree in booster.dump_model()['tree_info']:
        visit(tree['tree_structure'])
    for feature in levels:
        for value in (None, -1, len(levels[feature]) + 100):
            probes.append({**anchor, feature: value})
    probe_frame = pl.DataFrame(probes or [anchor], schema=training_codes.schema)
    evidence = {}
    selected_model = None
    for mode in ('analysis', 'max'):
        started = time.perf_counter()
        converted = from_booster(booster, holdout_codes, consolidation=mode)
        conversion_and_parity_seconds = time.perf_counter() - started
        boundary = from_booster(booster, probe_frame, consolidation=mode)
        if converted.parity['status'] != 'passed' or boundary.parity['status'] != 'passed':
            raise RuntimeError({'holdout': converted.parity, 'boundary': boundary.parity})
        model = (converted.model.with_categories(levels)
                 .with_response('avenue_frequency', exposure='exposure', exposure_role='weight'))
        converted.model = model
        converted.save(out / mode)
        loaded = Workbook.load_csv_dir(str(out / mode)).to_model()
        quotes = holdout.select(FEATURES)
        reference = booster.predict(holdout_codes.to_numpy())
        np.testing.assert_allclose(loaded.predict(quotes).to_series(), reference, rtol=1e-12, atol=1e-12)
        tables = model.to_workbook(scale='factor').tables
        times = []
        for _ in range(3):
            started = time.perf_counter()
            loaded.predict(quotes)
            times.append(time.perf_counter() - started)
        quote_times = []
        for _ in range(30):
            started = time.perf_counter()
            loaded.predict(quotes.head(1))
            quote_times.append(time.perf_counter() - started)
        evidence[mode] = {'holdout_parity': converted.parity, 'boundary_parity': boundary.parity,
                         'conversion_and_parity_seconds': conversion_and_parity_seconds,
                         'tables': len(tables), 'total_rows': sum(t.height for t in tables),
                         'largest_table': max(t.height for t in tables),
                         'largest_interaction_order': max([t.width-1 for t in tables[1:]] or [0]),
                         'warm_batch_seconds': times, 'warm_batch_median_seconds': statistics.median(times),
                         'single_quote_median_seconds': statistics.median(quote_times)}
        if mode == 'max':
            selected_model = model
    y, w = holdout['avenue_frequency'].to_numpy(), holdout['exposure'].to_numpy()
    comparison = pl.DataFrame([{'model': name, 'mean_poisson_deviance': mean_poisson_deviance(
        y, model.predict(holdout).to_series().to_numpy(), sample_weight=w)}
        for name, model in [('glm_frequency', baseline), ('booster_frequency', selected_model)]])
    comparison.write_csv(out / 'frequency_comparison.csv')
    selected_model.explain(holdout.select(FEATURES).head(20))['contributions'].write_csv(out / 'quote_explanations.csv')
    edited_dir = out / 'edited'
    selected_model.to_workbook().save_csv_dir(str(edited_dir))
    intercept = next(edited_dir.glob('*intercept.csv'))
    pl.read_csv(intercept).with_columns((pl.col('Relativity') * 1.05).alias('Relativity')).write_csv(intercept)
    edited = Workbook.load_csv_dir(str(edited_dir)).to_model()
    # Reconcile every quote numerically, then retain detailed attribution for a sample.
    np.testing.assert_allclose(edited.predict(holdout.select(FEATURES)).to_series(),
                               selected_model.predict(holdout.select(FEATURES)).to_series() * 1.05,
                               rtol=1e-12, atol=1e-12)
    (out / 'edited_report.md').write_text(edited.report(holdout).markdown)
    result = {'module': module_name, 'version': lgb.__version__, 'device': 'cpu',
              'tuned_interaction_penalties': tuning.tuned_interaction_penalties,
              'tuning_seconds': tuning_seconds, 'final_fit_seconds': fit_seconds,
              'selected_trial': asdict(selected), 'modes': evidence,
              'booster_convergence': 'finite schedule completed; no GLM score-convergence certificate',
              'limitations': ['four-trial study; not an exhaustive predictive search',
                              'inner LightGBM CV shares training Dataset bins; final holdout excluded',
                              'timings are local observations, not isolated comparative benchmarks']}
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return selected_model
