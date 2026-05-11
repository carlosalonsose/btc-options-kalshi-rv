import math
import unittest

import numpy as np
import pandas as pd

from src.model.black_scholes import bs_call, bs_put, implied_vol
from src.model.fees import kalshi_fee, total_fee_sell_yes
from src.model.intraday_q import range_prob_open_ended
from src.model.replication_discrete import replicate_bin
from src.model.svi import SVIParams


class BlackScholesTests(unittest.TestCase):
    def test_put_call_parity_at_zero_rate(self):
        spot = 100.0
        strike = 105.0
        maturity = 0.25
        sigma = 0.60

        call = bs_call(spot, strike, maturity, sigma)
        put = bs_put(spot, strike, maturity, sigma)

        self.assertAlmostEqual(call - put, spot - strike, places=10)

    def test_implied_vol_recovers_input_sigma(self):
        spot = 100.0
        strike = 100.0
        maturity = 0.10
        sigma = 0.75
        price = bs_call(spot, strike, maturity, sigma)

        recovered = implied_vol(price, spot, strike, maturity, "C")

        self.assertTrue(math.isfinite(recovered))
        self.assertAlmostEqual(recovered, sigma, places=6)


class FeeTests(unittest.TestCase):
    def test_kalshi_fee_rounds_up_to_cent(self):
        self.assertEqual(kalshi_fee(0.50), 0.02)
        self.assertEqual(kalshi_fee(0.01), 0.01)
        self.assertEqual(kalshi_fee(0.0), 0.0)
        self.assertEqual(kalshi_fee(1.0), 0.0)

    def test_total_sell_fee_includes_deribit_constant(self):
        self.assertAlmostEqual(total_fee_sell_yes(0.50, deribit_fee_const=0.015), 0.035)


class ProbabilityTests(unittest.TestCase):
    def test_open_ended_bins_sum_to_one(self):
        params = SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.1)
        kwargs = dict(F=100.0, T_K=1.0 / 365.0, T_D=1.0, p=params)
        bins = [
            (None, 90.0),
            (90.0, 100.0),
            (100.0, 110.0),
            (110.0, None),
        ]

        total = sum(range_prob_open_ended(lo, hi, **kwargs) for lo, hi in bins)

        self.assertAlmostEqual(total, 1.0, places=10)


class DiscreteReplicationTests(unittest.TestCase):
    def test_replicate_bin_uses_bracketing_call_vertical(self):
        calls = pd.DataFrame(
            {
                "strike": [100.0, 200.0],
                "bid_price": [10.0, 4.0],
                "ask_price": [12.0, 5.0],
            }
        )

        result = replicate_bin(120.0, 180.0, calls)

        self.assertIsNotNone(result)
        self.assertEqual(result["K_low"], 100.0)
        self.assertEqual(result["K_high"], 200.0)
        self.assertAlmostEqual(result["width_kalshi"], 60.0)
        self.assertAlmostEqual(result["vertical_ask_cost"], 8.0)
        self.assertAlmostEqual(result["vertical_bid_cost"], 5.0)
        self.assertAlmostEqual(result["q_buy_repl_disc"], 0.048)
        self.assertAlmostEqual(result["q_sell_repl_disc"], 0.030)


if __name__ == "__main__":
    unittest.main()
