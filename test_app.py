import unittest
from app import app

class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_and_health(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'not a verdict', response.data)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(self.client.get('/health').json['status'], 'ok')

    def test_valid(self):
        response = self.client.post('/api/analyze', json={'text': 'I spent the afternoon learning something new and making plans for the weekend with my friends.'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json['scores']), 3)
        self.assertAlmostEqual(sum(x['score'] for x in response.json['scores']), 1, places=3)
        self.assertNotIn('text', response.json)

    def test_bad_input(self):
        for value in ['', 123, None, 'a'*3001]:
            self.assertEqual(self.client.post('/api/analyze', json={'text': value}).status_code, 400)
        self.assertEqual(self.client.post('/api/analyze', json=[]).status_code, 400)
        self.assertEqual(self.client.post('/api/analyze', data='x'*21000).status_code, 413)
        self.assertEqual(self.client.post('/api/analyze', json={'text': 'qxzx '*20}).status_code, 422)

    def test_cross_site(self):
        self.assertEqual(self.client.post('/api/analyze', json={'text': 'testing'}, headers={'Sec-Fetch-Site':'cross-site'}).status_code, 403)

if __name__ == '__main__':
    unittest.main()
