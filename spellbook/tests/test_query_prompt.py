import os
import sys
import unittest


_SPELLBOOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SPELLBOOK)

from utils.query_prompt import window_position  # noqa: E402


class PromptGeometryTests(unittest.TestCase):
    def test_prompt_centers_on_viewer(self):
        self.assertEqual(window_position((440, 0, 800, 900), 480, 116), (600, 392))
        self.assertEqual(window_position(None), (100, 100))


if __name__ == "__main__":
    unittest.main()
