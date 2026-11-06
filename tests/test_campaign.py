import unittest

import pandas as pd

from campaign_baseline import CATEGORICAL, NUMERIC, evaluate


def sample():
    data = {col: [1, 2, 3, 4] * 5 for col in NUMERIC}
    data.update({col: ["YES", "NO", "YES", "NO"] * 5 for col in CATEGORICAL})
    data["pep"] = ["YES", "NO", "YES", "NO"] * 5
    return pd.DataFrame(data)


class CampaignTests(unittest.TestCase):
    def test_pipeline_and_accounting(self):
        data = sample()
        test = data.copy()
        test["region"] = "previously unseen"
        result = evaluate(data, test)
        self.assertAlmostEqual(result["threshold"], 70 / 170)
        self.assertEqual(result["test_rows"], 20)
        self.assertEqual(result["observed_profit"],
                         result["positive_targets"] * 100 - (result["targeted"] - result["positive_targets"]) * 70)
        self.assertEqual(result, evaluate(data, test))

    def test_invalid_input(self):
        data = sample()
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            evaluate(data.drop(columns="age"), data)
        with self.assertRaisesRegex(ValueError, "positive"):
            evaluate(data, data, benefit=0)
        data["pep"] = "YES"
        with self.assertRaisesRegex(ValueError, "both classes"):
            evaluate(data, data)


if __name__ == "__main__":
    unittest.main()
