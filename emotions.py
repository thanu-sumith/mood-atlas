"""Emotion suggestions and a separate, explicit-language tiredness cue."""
import re
DISPLAY = {'joy':'Happy / joyful','excitement':'Excited / enthusiastic','sadness':'Sad','neutral':'Neutral wording'}

def tiredness_cue(text):
    # A transparent phrase cue, not a learned emotion score or fatigue diagnosis.
    clauses = re.split(r'[.!?;,]|\bbut\b', text.lower().replace('’', "'"))
    for clause in clauses:
        if re.search(r"\b(not|never|isn't|aren't|wasn't|don't|doesn't|no longer)\b", clause):
            continue
        if re.search(r"\b(i am|i'm|i feel|feeling|he is|he's|she is|she's|they are|they're|we are|we're)\s+(?:so |very |really |completely |physically |a little )?(tired|sleepy|exhausted|worn out)\b(?!\s+of\b)", clause):
            return {'label':'Tired / low energy','source':'explicit wording',
                    'note':'The text explicitly describes tiredness; this is a phrase cue, not a trained prediction.'}
    return None

def analyze_emotions(text, model):
    features=model['vectorizer'].transform([text])
    if features.nnz == 0:
        return {'status':'uncertain','summary':'Not enough familiar English wording to interpret.', 'emotions':[], 'cues':[]}
    values=model['classifier'].predict_proba(features)[0]
    selected=[label for label, score, threshold in zip(model['labels'], values, model['thresholds']) if score >= threshold]
    expressive=[label for label in selected if label != 'neutral']
    cue=tiredness_cue(text)
    cues=[cue] if cue else []
    if expressive:
        status='emotion'
        summary='Possible emotions expressed in this text. Mixed emotions can appear together.'
        names=expressive
    elif cue:
        status='explicit_cue'
        summary='The text describes tiredness. No other emotion passed the model threshold.'
        names=[]
    elif 'neutral' in selected:
        status='neutral'
        summary='Neutral wording. The sentence does not establish how anyone feels.'
        names=['neutral']
    else:
        status='uncertain'
        summary='No clear emotion detected. More context may help; this does not mean the person feels neutral.'
        names=[]
    return {'status':status,'summary':summary,'emotions':[{'label':DISPLAY.get(n,n.capitalize()),'category':n} for n in names], 'cues':cues}
