import unittest
from src.app import app

class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"ScholarAI", res.data)
        self.assertIn(b"Upload Research Paper", res.data)

    def test_qa_page(self):
        res = self.client.get("/qa")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Ask the Paper", res.data)

    def test_compare_page(self):
        res = self.client.get("/compare")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Comparative Literature Study", res.data)

    def test_improve_page(self):
        res = self.client.get("/improve")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Publication Critique", res.data)

    def test_download_page(self):
        res = self.client.get("/download")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Export &amp; Download Center", res.data)

    def test_ask_empty_query(self):
        res = self.client.post("/ask", json={"query": ""})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    def test_upload_missing_file(self):
        res = self.client.post("/upload")
        self.assertEqual(res.status_code, 400)

if __name__ == "__main__":
    unittest.main()
