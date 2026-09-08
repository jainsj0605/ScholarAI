import unittest
from src.rag import chunk_text, VectorStore

class TestRAGComponents(unittest.TestCase):
    def test_chunk_text_basic(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunk_text(text, size=10)
        self.assertEqual(chunks, ["abcdefghij", "klmnopqrst", "uvwxyz"])

    def test_chunk_text_empty(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text(None), [])

    def test_vector_store_indexing_and_retrieval(self):
        store = VectorStore()
        docs = [
            "Convolutional Neural Networks are designed for computer vision and image processing.",
            "Transformer architectures rely on self-attention mechanisms for language modeling.",
            "Reinforcement learning optimizes decision-making policies through reward signals."
        ]
        store.add_chunks(docs)
        self.assertEqual(len(store.documents), 3)

        # Retrieve for vision query
        results = store.retrieve("computer vision images", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Convolutional", results[0])

        # Retrieve for language query
        results = store.retrieve("self-attention transformers", k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Transformer", results[0])

        # Clear store
        store.clear()
        self.assertEqual(len(store.documents), 0)
        self.assertEqual(store.retrieve("anything"), [])

if __name__ == "__main__":
    unittest.main()
