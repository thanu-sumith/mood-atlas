import json, re
import time
from collections import deque
from threading import Lock
from pathlib import Path
import joblib
from emotions import analyze_emotions
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).parent
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20000
model = joblib.load(ROOT / 'emotion_model.joblib')
metrics = json.loads((ROOT / 'emotion_metrics.json').read_text())
recent_requests = deque()
request_lock = Lock()

@app.after_request
def headers(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    return response

@app.get('/')
def home():
    return render_template('index.html', metrics=metrics)

@app.get('/health')
def health():
    return jsonify(status='ok', model_loaded=True, model_version=metrics['model_version'])

@app.errorhandler(413)
def oversized(error):
    return jsonify(error='Please use fewer than 3,000 characters.'), 413

@app.post('/api/analyze')
def analyze():
    if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
        return jsonify(error='Please use fewer than 3,000 characters.'), 413
    if request.headers.get('Sec-Fetch-Site') == 'cross-site':
        return jsonify(error='Please submit from this website.'), 403
    with request_lock:
        now = time.monotonic()
        while recent_requests and recent_requests[0] < now - 60:
            recent_requests.popleft()
        if len(recent_requests) >= 120:
            return jsonify(error='The demo is busy. Please try again in a minute.'), 429
        recent_requests.append(now)
    data = request.get_json(silent=True)
    value = data.get('text') if isinstance(data, dict) else None
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 3000:
        return jsonify(error='Enter between 3 and 3,000 characters.'), 400
    text = re.sub(r'\s+', ' ', re.sub(r'https?://\S+', '', value.lower())).strip()
    return jsonify(**analyze_emotions(text, model), model_version=metrics['model_version'])
