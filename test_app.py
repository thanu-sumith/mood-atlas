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
        self.assertIn(response.json['status'], ['neutral', 'emotion', 'uncertain', 'explicit_cue'])
        self.assertNotIn('scores', response.json)
        self.assertEqual(response.json['model_version'], 'goemotions-v1')
        self.assertNotIn('text', response.json)

    def test_bad_input(self):
        for value in ['', 123, None, 'a'*3001]:
            self.assertEqual(self.client.post('/api/analyze', json={'text': value}).status_code, 400)
        self.assertEqual(self.client.post('/api/analyze', json=[]).status_code, 400)
        self.assertEqual(self.client.post('/api/analyze', data='x'*21000).status_code, 413)
        self.assertEqual(self.client.post('/api/analyze', json={'text': 'qxzx '*20}).json['status'], 'uncertain')

    def test_cricket_does_not_imply_emotion_or_health_condition(self):
        result = self.client.post('/api/analyze', json={'text': 'a man playing cricket in the ground'}).json
        self.assertIn(result['status'], ['neutral', 'uncertain'])
        self.assertNotIn('scores', result)
        self.assertFalse(any(x['category'] in ['depression', 'suicide'] for x in result['emotions']))

    def test_independent_positive_emotions(self):
        result = self.client.post('/api/analyze', json={'text': 'I am so happy we won the cricket match! I am excited about the next game.'}).json
        self.assertTrue({'joy', 'excitement'}.issubset({x['category'] for x in result['emotions']}))

    def test_sadness(self):
        result = self.client.post('/api/analyze', json={'text': 'I am sad because I miss my friend.'}).json
        self.assertIn('sadness', [x['category'] for x in result['emotions']])

    def test_tiredness_is_separate_and_negation_aware(self):
        for text, expected in [('I am tired after playing cricket.', True), ('I am not tired after playing cricket.', False), ('I am tired of this argument.', False)]:
            result = self.client.post('/api/analyze', json={'text': text}).json
            self.assertEqual(bool(result['cues']), expected)
            for cue in result['cues']:
                self.assertEqual(cue['source'], 'explicit wording')
                self.assertNotIn('score', cue)

    def test_cross_site(self):
        self.assertEqual(self.client.post('/api/analyze', json={'text': 'testing'}, headers={'Sec-Fetch-Site':'cross-site'}).status_code, 403)

if __name__ == '__main__':
    unittest.main()
