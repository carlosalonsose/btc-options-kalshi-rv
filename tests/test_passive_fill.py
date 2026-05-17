import unittest

import numpy as np
import pandas as pd

from src.runner.passive_fill import _fill_side


def _two_snap(yes_bid_t, yes_ask_t, yes_bid_n, yes_ask_n,
              q_deribit=0.30, q_buy_repl=0.30, q_sell_repl=0.30,
              outcome=0):
    """A single ticker observed at two consecutive snapshots."""
    return pd.DataFrame({
        "ticker": ["A", "A"],
        "event_ticker": ["E", "E"],
        "snap_ts": pd.to_datetime(["2026-05-12T10:00:00Z",
                                   "2026-05-12T10:00:10Z"]),
        "q_deribit": [q_deribit, q_deribit],
        "yes_bid": [yes_bid_t, yes_bid_n],
        "yes_ask": [yes_ask_t, yes_ask_n],
        "q_buy_repl_disc": [q_buy_repl, q_buy_repl],
        "q_sell_repl_disc": [q_sell_repl, q_sell_repl],
        "outcome": [outcome, outcome],
        "T_K": [0.001, 0.001],
    })


class PassiveFillTests(unittest.TestCase):
    def test_adverse_fills_only_when_market_trades_through(self):
        # fair 0.30, h 0.01 -> our_ask 0.31, posted inside book [0.20, 0.45].
        # next snapshot bid jumps to 0.40 >= 0.31 -> SELL fills.
        df = _two_snap(yes_bid_t=0.20, yes_ask_t=0.45,
                       yes_bid_n=0.40, yes_ask_n=0.50)
        tr = _fill_side(df, h=0.01, deribit_fee=0.015, adverse=True)
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr.iloc[0]["side"], "sell")

    def test_adverse_no_fill_when_market_does_not_move(self):
        # next snapshot unchanged -> resting quote never trades through.
        df = _two_snap(yes_bid_t=0.20, yes_ask_t=0.45,
                       yes_bid_n=0.20, yes_ask_n=0.45)
        tr = _fill_side(df, h=0.01, deribit_fee=0.015, adverse=True)
        self.assertTrue(tr.empty)

    def test_naive_control_fills_every_posted_quote(self):
        # control assumes the posted quote always fills; both sides postable.
        df = _two_snap(yes_bid_t=0.20, yes_ask_t=0.45,
                       yes_bid_n=0.20, yes_ask_n=0.45)
        tr = _fill_side(df, h=0.01, deribit_fee=0.015, adverse=False)
        self.assertGreaterEqual(len(tr), 1)

    def test_unhedgeable_bin_is_not_posted(self):
        df = _two_snap(yes_bid_t=0.20, yes_ask_t=0.45,
                       yes_bid_n=0.40, yes_ask_n=0.50,
                       q_buy_repl=np.nan, q_sell_repl=np.nan)
        tr = _fill_side(df, h=0.01, deribit_fee=0.015, adverse=True)
        self.assertTrue(tr.empty)


if __name__ == "__main__":
    unittest.main()
