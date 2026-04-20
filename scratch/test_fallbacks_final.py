import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add the project directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
from config import TEXT_MODEL, FALLBACK_MODEL, FAST_MODEL

class TestFallbackChain(unittest.TestCase):

    @patch('utils.client.chat.completions.create')
    def test_tier1_success(self, mock_create):
        # Mock success on first try
        mock_create.return_value.choices = [MagicMock(message=MagicMock(content="Tier 1 Output"))]
        
        result = utils.llm("test prompt")
        self.assertEqual(result, "Tier 1 Output")
        self.assertEqual(mock_create.call_count, 1)

    @patch('utils.client.chat.completions.create')
    def test_fallback_to_tier2(self, mock_create):
        # Mock failure on Tier 1 (429), success on Tier 2
        mock_create.side_effect = [
            Exception("Error code: 429 - Rate limit reached for Tier 1"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Tier 2 Output"))])
        ]
        
        result = utils.llm("test prompt")
        self.assertEqual(result, "Tier 2 Output")
        self.assertEqual(mock_create.call_count, 2)
        # Verify it used Tier 2
        self.assertEqual(mock_create.call_args_list[1][1]['model'], FALLBACK_MODEL)

    @patch('utils.client.chat.completions.create')
    def test_fallback_to_tier3_daily_limit(self, mock_create):
        # Mock daily limit on Tier 1 & 2, success on Tier 3
        mock_create.side_effect = [
            Exception("Error code: 429 - tokens per day limit reached for Tier 1"),
            Exception("Error code: 429 - tokens per day limit reached for Tier 2"),
            MagicMock(choices=[MagicMock(message=MagicMock(content="Tier 3 Output"))])
        ]
        
        result = utils.llm("test prompt")
        self.assertEqual(result, "Tier 3 Output")
        self.assertEqual(mock_create.call_count, 3)
        # Verify it used Tier 3 (8B)
        self.assertEqual(mock_create.call_args_list[2][1]['model'], FAST_MODEL)

    @patch('utils.client.chat.completions.create')
    def test_all_tiers_exhausted(self, mock_create):
        # Mock failure on all 3 tiers
        mock_create.side_effect = [
            Exception("429 - tokens per day limit Tier 1"),
            Exception("429 - tokens per day limit Tier 2"),
            Exception("429 - tokens per day limit Tier 3")
        ]
        
        result = utils.llm("test prompt")
        self.assertIn("CRITICAL: All AI models have reached their daily token limits", result)
        self.assertEqual(mock_create.call_count, 3)

    @patch('utils.client.chat.completions.create')
    def test_specific_model_no_fallback(self, mock_create):
        # If I request FAST_MODEL directly, it should NOT fall back to others if it fails
        mock_create.side_effect = Exception("429 - tokens per day limit Tier 3")
        
        result = utils.llm("test prompt", model=FAST_MODEL)
        self.assertIn("Error:", result)
        self.assertNotIn("CRITICAL", result) # because we didn't try the fallback list
        self.assertEqual(mock_create.call_count, 1)

if __name__ == "__main__":
    unittest.main()
