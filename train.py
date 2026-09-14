"""Rebuild the original text-classification approach; never export raw posts."""
import csv, hashlib, json, re, sys
from collections import Counter
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent
csv.field_size_limit(10_000_000)
rows, conflicts, total, invalid = {}, set(), 0, 0
with open(sys.argv[1], encoding='utf-8-sig', newline='') as source:
    for row in csv.DictReader(source):
        total += 1
        text = re.sub(r'\s+', ' ', re.sub(r'https?://\S+', '', row.get('text', '').lower())).strip()
        label = row.get('label', '').strip().lower()
        if len(text) < 10 or label not in {'suicide', 'depression', 'non-suicide'}:
            invalid += 1
            continue
        key = hashlib.sha256(text.encode()).hexdigest()
        if key in rows and rows[key][1] != label:
            conflicts.add(key)
        rows[key] = (text, label)
clean = [row for key, row in rows.items() if key not in conflicts]
x, y = zip(*clean)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=40000, min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(x_train, y_train)
pred = model.predict(x_test)
majority = Counter(y_train).most_common(1)[0][0]
report = {'source_rows': total, 'usable_unique_rows': len(clean), 'invalid_rows': invalid,
          'conflicting_texts_excluded': len(conflicts), 'train_rows': len(x_train), 'test_rows': len(x_test),
          'class_counts': dict(Counter(y)), 'accuracy': accuracy_score(y_test, pred),
          'majority_baseline': sum(v == majority for v in y_test) / len(y_test),
          'report': classification_report(y_test, pred, output_dict=True, zero_division=0),
          'classes': list(model.classes_), 'confusion_matrix': confusion_matrix(y_test, pred, labels=model.classes_).tolist(),
          'method': 'Normalized exact deduplication before stratified 80/20 split, seed 42; training-only TF-IDF (40k features, 1–2 grams) + logistic regression. Negations retained. No external clinical validation; near-duplicates and author overlap may remain.'}
joblib.dump(model, ROOT / 'model.joblib', compress=3)
(ROOT / 'metrics.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
