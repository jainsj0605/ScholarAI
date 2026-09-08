import unittest
from src.search import clean_query, normalize_venue, _engineering_bonus

class TestSearchUtilities(unittest.TestCase):
    def test_clean_query_removes_preambles(self):
        self.assertEqual(clean_query("Keywords: Machine Learning, NLP"), "Machine Learning, NLP")
        self.assertEqual(clean_query("topics: Quantum Computing"), "Quantum Computing")
        self.assertEqual(clean_query("query: deep learning"), "deep learning")

    def test_clean_query_removes_quotes_and_brackets(self):
        self.assertEqual(clean_query('“Transformer” [Models]'), "Transformer Models")
        self.assertEqual(clean_query('(Graph) {Neural} Networks'), "Graph Neural Networks")

    def test_clean_query_empty_input(self):
        self.assertEqual(clean_query(""), "")
        self.assertEqual(clean_query(None), "")

    def test_normalize_venue(self):
        self.assertEqual(
            normalize_venue("Institute of Electrical and Electronics Engineers Transactions"),
            "IEEE"
        )
        self.assertEqual(normalize_venue(""), "Academic Source")
        self.assertEqual(normalize_venue(None), "Academic Source")
        self.assertEqual(normalize_venue("ACM SIGCOMM"), "ACM")
        self.assertEqual(normalize_venue("Elsevier Computer Communications"), "Elsevier")
        self.assertEqual(
            normalize_venue("Some Very Long Unknown University Journal Press Title"),
            "Some Very Long Unknown Univ..."
        )

    def test_engineering_bonus(self):
        self.assertEqual(_engineering_bonus({"venue": "IEEE"}), 0.10)
        self.assertEqual(_engineering_bonus({"venue": "Springer"}), 0.10)
        self.assertEqual(_engineering_bonus({"venue": "General Blog"}), 0.0)

if __name__ == "__main__":
    unittest.main()
