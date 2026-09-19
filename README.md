# Mood Atlas

An English text-emotion demo by Sumith Reddy. The original three-way classifier forced all text into depression, suicide, or non-suicide labels. This revision replaces that inference path with a multilabel everyday-emotion baseline and removes mental-health percentages.

## What changed

- Trained on Google's **GoEmotions**: 27 emotion categories plus neutral.
- Joy displays as **Happy / joyful** and excitement as **Excited / enthusiastic**. Other categories include sadness, anger, gratitude, love, fear, curiosity, surprise and optimism.
- Labels are predicted independently, so multiple emotions can appear together. A validation threshold must be met; no forced top label or 100% pie chart.
- Ambiguous text produces **No clear emotion detected**. Neutral wording does not establish someone's actual feelings.
- **Tired / low energy** is a separate, explicitly labelled phrase cue. It checks direct wording, skips common negations and “tired of” expressions, and has no numeric confidence. It is not a trained fatigue model; idioms, quotation and complex negation remain limitations.
- The old `model.joblib`, `metrics.json`, and `train.py` are historical research artifacts; the application no longer loads them.

## Evidence and limits

Official splits after normalized exact deduplication: 43,189 train, 5,375 validation, 5,379 test examples. Training-only TF-IDF (50,000 word/unigram-bigram features) and one-vs-rest logistic regression. Validation selects per-label F0.5 thresholds requiring >=60% precision and at least five predictions. Labels without qualifying thresholds abstain. These are validation choices, not guaranteed production precision.

Untouched test set: micro precision **0.673**, recall **0.389**, F1 **0.493**. At least one label is emitted on **62.4%** of test examples. Full per-label results and source file hashes are in `emotion_metrics.json`. Metrics cover the learned model only, not the tiredness rule or UI neutral-suppression policy.

This is a lightweight baseline, not proof of improved general accuracy: the old and new models solve different tasks on different data. Reddit language differs from everyday user prompts. Sarcasm, negation, implicit emotions and paraphrases can fail. For example, an explicit “enthusiastic” sentence may still abstain, mixed happy/sad text can miss one emotion, and exhausted wording can incorrectly also suggest sadness. No model score diagnoses depression or assesses suicide risk. A next precision improvement requires a separately annotated evaluation set from the demo's target domain, then a contextual encoder comparison or fine-tuning; do not invent training labels or dilute old probabilities.

## Reproduce and run

Python 3.12:

```sh
pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python prepare_model.py
python -m unittest -v
gunicorn app:app --bind 0.0.0.0:8000
```

The preparation command downloads the four public GoEmotions split/label files into a temporary directory, verifies committed SHA256 values, builds `emotion_model.joblib`, and deletes the raw training download. It fails if upstream data changes. No user submissions are used for training. Builds require network access to raw.githubusercontent.com; no inference API or GPU is required.

For the existing Render service, set its build command to `pip install -r requirements.txt && python prepare_model.py` (also declared in render.yaml). Existing dashboard services may not automatically adopt Blueprint edits. Keep the current start command and free plan. Verify `/health` reports `goemotions-v1` before considering deployment complete. Tests exercise neutral cricket wording, positive mixed emotions, sadness, tiredness/negation, request validation and security headers.

`POST /api/analyze` accepts JSON `{"text":"..."}` (3–3000 characters) and returns status, summary, emotions, cues and model_version. This replaces the old `scores` response. Frontend and API must deploy together. Requests are not intentionally stored or logged by the app. Existing rate limiting is process-local.

## Dataset attribution

[GoEmotions, Google Research](https://github.com/google-research/google-research/tree/master/goemotions), Demszky et al. (2020), *GoEmotions: A Dataset of Fine-Grained Emotions*, ACL. Dataset distributed under Apache 2.0; see [upstream license](https://github.com/google-research/google-research/blob/master/LICENSE). No raw Reddit posts are committed in this revision.
