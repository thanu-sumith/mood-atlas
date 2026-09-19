"""Train a CPU-friendly multilabel GoEmotions baseline; no raw posts are exported.
Usage: python train_emotions.py /path/to/goemotions/data
Download train.tsv, dev.tsv, test.tsv, emotions.txt from Google's goemotions/data.
"""
import csv, hashlib, json, sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).parent
source = Path(sys.argv[1])
labels = (source / 'emotions.txt').read_text().splitlines()
seen = set()
def read_split(name):
    texts, targets = [], []
    for text, ids, _ in csv.reader((source / name).open(), delimiter='\t'):
        key = ' '.join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
        targets.append([int(i) for i in ids.split(',')])
    return texts, MultiLabelBinarizer(classes=list(range(len(labels)))).fit_transform(targets)
train, y_train = read_split('train.tsv')
dev, y_dev = read_split('dev.tsv')
test, y_test = read_split('test.tsv')
vectorizer = TfidfVectorizer(max_features=50000, min_df=2, ngram_range=(1,2), sublinear_tf=True)
classifier = OneVsRestClassifier(LogisticRegression(C=4, max_iter=500, solver='liblinear', random_state=42))
classifier.fit(vectorizer.fit_transform(train), y_train)
prob = classifier.predict_proba(vectorizer.transform(dev))
thresholds=[]
for j in range(len(labels)):
    best, threshold = -1, 1.0
    for t in np.arange(.15, .91, .025):
        pred = prob[:,j] >= t
        tp = int((pred & (y_dev[:,j] == 1)).sum())
        precision = tp / max(1,int(pred.sum()))
        recall = tp / max(1,int(y_dev[:,j].sum()))
        f05 = 1.25*precision*recall/max(1e-9,.25*precision+recall)
        if precision >= .6 and pred.sum() >= 5 and f05 > best:
            best, threshold = f05, float(t)
    thresholds.append(threshold)
pred = classifier.predict_proba(vectorizer.transform(test)) >= thresholds
report=classification_report(y_test,pred,target_names=labels,output_dict=True,zero_division=0)
metrics={'model_version':'goemotions-v1','train_rows':len(train),'validation_rows':len(dev),'test_rows':len(test),
 'classes':labels,'thresholds':dict(zip(labels,thresholds)), 'report':report,
 'micro_f1':f1_score(y_test,pred,average='micro',zero_division=0),
 'micro_precision':precision_score(y_test,pred,average='micro',zero_division=0),
 'micro_recall':recall_score(y_test,pred,average='micro',zero_division=0),
 'coverage':float(pred.any(axis=1).mean()),
 'source':'https://github.com/google-research/google-research/tree/master/goemotions',
 'source_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ['train.tsv','dev.tsv','test.tsv','emotions.txt']},
 'method':'GoEmotions official train/dev/test splits; normalized exact duplicates excluded across splits, training first. Training-only TF-IDF and one-vs-rest logistic regression. Per-label thresholds selected on validation F0.5 with precision >= 0.60 and >= 5 predictions; unsupported labels abstain. Test set untouched during fitting and threshold selection. English Reddit domain; not clinical or real-world accuracy.'}
joblib.dump({'vectorizer':vectorizer,'classifier':classifier,'labels':labels,'thresholds':thresholds},ROOT/'emotion_model.joblib',compress=3)
(ROOT/'emotion_metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps({k:metrics[k] for k in ['train_rows','validation_rows','test_rows','micro_f1','micro_precision','micro_recall','coverage']},indent=2))
