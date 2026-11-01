import tempfile, unittest
from pathlib import Path
from helpers import *
from planner.data import oulad, synthetic
from planner.data.profiles import make_profiles

FORBIDDEN = {"gender", "age_band", "region", "disability", "imd_band", "highest_education"}

class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        synthetic.generate(cls.dir, seed=1, n_students=60)
        oulad.check_schema(cls.dir)
        cls.df, cls.meta = oulad.finalize(oulad.build_assessment_table(cls.dir))

    def test_normal_features_present_no_nan_no_demographics(self):
        self.assertTrue(set(C.FEATURES + [C.TARGET]) <= set(self.df.columns))
        self.assertEqual(len(C.FEATURES), 12)
        self.assertFalse(self.df[C.FEATURES + [C.TARGET]].isna().any().any())
        self.assertFalse(FORBIDDEN & set(C.FEATURES))

    def test_normal_scores_in_range(self):
        self.assertTrue(self.df.score.between(0, 100).all())

    def test_normal_reproducible_with_seed(self):
        d2 = Path(tempfile.mkdtemp()); synthetic.generate(d2, seed=1, n_students=60)
        df2, _ = oulad.finalize(oulad.build_assessment_table(d2))
        self.assertTrue(self.df.equals(df2))

    def test_normal_profiles_at_least_requested(self):
        self.assertGreaterEqual(len(make_profiles(self.df, 30, seed=3)), 20)

    def test_edge_missing_file_gives_clear_error(self):
        with self.assertRaises(FileNotFoundError): oulad.check_schema(Path(tempfile.mkdtemp()))

    def test_edge_clicks_per_session_positive(self):
        self.assertGreaterEqual(self.meta["clicks_per_session"], 1)

if __name__ == "__main__":
    unittest.main()
