import os
import joblib
import numpy as np

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']

def load_ag_news():
    ds = load_dataset('ag_news')

    # ds['train'] has "text" and "label"
    X = np.array(ds['train']['text'])
    y = np.array(ds['train']['label'])

    return X, y

def build_pipeline():
    # Strong baseline for text classification
    return Pipeline(steps = [
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1,2),
            min_df=2,
            max_df=0.95
        )),
        ('clf', LogisticRegression(
            max_iter=2000,
            n_jobs=None,
            C=2.0
        ))
    ])

def main():
    X, y = load_ag_news()

    # Holdout split for fast iteration
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average='macro')

    print('========== Holdout Results ============')
    print(f'Accuracy: {acc:.4f}')
    print(f'Macro F1: {macro_f1:.4f}\n')
    print(classification_report(y_test, preds, target_names=LABELS))

    os.makedirs('models', exist_ok=True)
    joblib.dump(pipe, 'models/text_clf.joblib')
    print('Saved model -> models/text_clf.joblib')

if __name__ == '__main__':
    main()