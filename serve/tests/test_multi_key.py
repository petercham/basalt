"""Several API keys on one server: a team keeps its own key each.

Several keys on one server: a team keeps its own key each, and the ledger still names each caller.
The engine takes keys separated by commas ("k1,k2") or as a list; serve accepts the same form, so a
key file copied between the two keeps working.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from serve.security import api_key_of, exposed_without_key, key_list, key_matches  # noqa: E402


class SeveralKeys(unittest.TestCase):
    def test_a_comma_separated_string_names_every_key(self):
        self.assertEqual(key_list("k1,k2"), ["k1", "k2"])
        self.assertEqual(key_list(" k1 , k2 , "), ["k1", "k2"])
        self.assertEqual(key_list(["k1", "k2"]), ["k1", "k2"])
        self.assertEqual(key_list(""), [])
        self.assertEqual(key_list(None), [])

    def test_any_of_the_keys_passes_and_a_stranger_does_not(self):
        keys = "k1,k2,k3"
        for k in ("k1", "k2", "k3"):
            self.assertTrue(key_matches(k, keys), k)
        self.assertFalse(key_matches("k4", keys))
        self.assertFalse(key_matches("k1 ", keys))
        self.assertFalse(key_matches("", keys))
        self.assertFalse(key_matches("k1x", keys))

    def test_a_list_of_keys_works_the_same(self):
        keys = ["k1", "k2"]
        self.assertTrue(key_matches("k2", keys))
        self.assertFalse(key_matches("k3", keys))

    def test_one_key_still_behaves_as_before(self):
        self.assertTrue(key_matches("s3cret", "s3cret"))
        self.assertFalse(key_matches("s3cret", "other"))
        self.assertEqual(api_key_of(" s3cret "), "s3cret")

    def test_a_key_that_is_all_blank_is_an_error_not_no_key(self):
        for blank in (" ", "\r\n", " , "):
            with self.assertRaises(ValueError):
                api_key_of(blank)

    def test_a_list_holding_a_blank_key_is_an_error(self):
        with self.assertRaises(ValueError):
            api_key_of(["k1", " "])

    def test_keys_on_a_network_listener_are_not_exposure(self):
        self.assertFalse(exposed_without_key("0.0.0.0", "k1,k2"))
        self.assertFalse(exposed_without_key("0.0.0.0", ["k1"]))
        self.assertTrue(exposed_without_key("0.0.0.0", ""))
        self.assertTrue(exposed_without_key("0.0.0.0", []))


if __name__ == "__main__":
    unittest.main()
