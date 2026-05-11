import unittest

from src.event_snapshotter import canonical_quote_state, changed_sides, side_hashes, state_hash


class EventSnapshotterTests(unittest.TestCase):
    def test_quote_state_hash_is_stable_to_market_order(self):
        payload_a = {
            "kalshi": {
                "markets_open": [
                    {"ticker": "B", "yes_bid_dollars": "0.10", "yes_ask_dollars": "0.20"},
                    {"ticker": "A", "yes_bid_dollars": "0.30", "yes_ask_dollars": "0.40"},
                ],
                "events_open": [],
            },
            "deribit": {},
        }
        payload_b = {
            "kalshi": {
                "markets_open": [
                    {"ticker": "A", "yes_bid_dollars": "0.30", "yes_ask_dollars": "0.40"},
                    {"ticker": "B", "yes_bid_dollars": "0.10", "yes_ask_dollars": "0.20"},
                ],
                "events_open": [],
            },
            "deribit": {},
        }

        self.assertEqual(
            state_hash(canonical_quote_state(payload_a, watch="kalshi")),
            state_hash(canonical_quote_state(payload_b, watch="kalshi")),
        )

    def test_quote_state_ignores_non_quote_fields(self):
        base = {
            "kalshi": {
                "markets_open": [
                    {
                        "ticker": "A",
                        "yes_bid_dollars": "0.30",
                        "yes_ask_dollars": "0.40",
                        "title": "human readable text",
                    }
                ],
                "events_open": [],
            },
            "deribit": {},
        }
        noisy = {
            "kalshi": {
                "markets_open": [
                    {
                        "ticker": "A",
                        "yes_bid_dollars": "0.30",
                        "yes_ask_dollars": "0.40",
                        "title": "changed text",
                    }
                ],
                "events_open": [],
            },
            "deribit": {},
        }

        self.assertEqual(
            state_hash(canonical_quote_state(base, watch="kalshi")),
            state_hash(canonical_quote_state(noisy, watch="kalshi")),
        )

    def test_changed_sides_reports_only_modified_side(self):
        first = canonical_quote_state(
            {
                "kalshi": {"markets_open": [{"ticker": "A", "yes_bid_dollars": "0.30"}], "events_open": []},
                "deribit": {"book_summary_usdc": [], "book_summary_btc": []},
            },
            watch="both",
        )
        second = canonical_quote_state(
            {
                "kalshi": {"markets_open": [{"ticker": "A", "yes_bid_dollars": "0.31"}], "events_open": []},
                "deribit": {"book_summary_usdc": [], "book_summary_btc": []},
            },
            watch="both",
        )

        self.assertEqual(changed_sides(side_hashes(first), side_hashes(second)), ["kalshi"])


if __name__ == "__main__":
    unittest.main()
