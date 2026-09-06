import unittest
import polars as pl
from avenue_model import Plan


class ReviewCalibrationTests(unittest.TestCase):
    def test_aggregate_drift_is_not_a_rebase_instruction_or_bucket_diagnosis(self):
        training = pl.DataFrame({'y': [1., 2.] * 20})
        model = Plan('poisson').fit(training, 'y')
        report = model.report(pl.DataFrame({'y': [3.] * 40}))
        findings = {f['code']: f['message'] for f in report.findings}
        drift = findings['calibration_drift']
        self.assertIn('2.0000', drift)
        self.assertIn('120.0000 actual against 60.0000 expected', drift)
        self.assertNotIn('100.00% less', drift)
        self.assertNotIn('Rebase', drift)
        buckets = findings['bucket_miscalibration']
        self.assertNotIn('calibrated in aggregate', buckets)
        self.assertIn('descriptive flags', buckets)
        self.assertIn('claim support', buckets)
        self.assertIn(f'of {report.validation.calibration.height} equal-exposure', buckets)
