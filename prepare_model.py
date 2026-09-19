"""Fetch hash-verified public training splits and rebuild the deployed artifact."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).parent
expected = json.loads((ROOT / 'emotion_metrics.json').read_text())['source_sha256']
with tempfile.TemporaryDirectory(prefix='goemotions-') as folder:
    for name, checksum in expected.items():
        path=Path(folder)/name
        urllib.request.urlretrieve('https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/'+name,path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise RuntimeError('Training source changed: '+name+'; review before rebuilding.')
    subprocess.run([sys.executable,str(ROOT/'train_emotions.py'),folder],check=True)
