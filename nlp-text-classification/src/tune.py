import os
import joblib
import numpy as np

from datasets import load_dataset
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, classification_report

LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']

def load_ag_news_train():
    ds = load_dataset('ag_news')

    X = np.array(ds['train']['text'])
    y = np.array(ds['train']['label'])

    return X, y


def build_pipeline():
    return Pipeline(steps = [
        ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
        ('clf', LinearSVC())
    ])


def main():
    X, y = load_ag_news_train()

    # Small holdout for sanity-check after CV picks best params
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y,
        test_size = 0.15,
        random_state=42,
        stratify=y
    )

    pipe = build_pipeline()

    # RandomizedSearchSV keeps this quick while still 'real'
    param_distributions = {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': [1, 2, 3, 5],
        'tfidf__max_df': [0.85, 0.9, 0.95, 1.0],
        'tfidf__sublinear_tf': [True, False],
        # Optional: if you want a robutness bump, try char n-grams in Steo 2
        # 'tfidf__analyzer': ['words'],
        'clf__C': [0.25, 0.5, 1.0, 2.0, 4.0]
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=18,
        scoring='f1_macro',
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print('\n===== Best CV Params =====')
    print(search.best_params_)
    print(f'Best CV macro F1: {search.best_score_:.4f}')

    # Holdout evaluation
    preds = best_model.predict(X_hold)
    acc = accuracy_score(y_hold, preds)
    macro_f1 = f1_score(y_hold, preds, average='macro')

    print('\n===== Holdout Results (post-tuning) =====')
    print(f'Accuracy: {acc:.4f}')
    print(f'Macro F1: {macro_f1:.4f}\n')
    print(classification_report(y_hold, preds, target_names=LABELS))

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/text_clf.joblib')
    joblib.dump(search.best_params_, 'models/best_params.joblib')
    print('\nSaved best model -> models/text_clf.joblib')
    print('Saved best params -> models/best_params.joblib')

if __name__ == '__main__':
    main()