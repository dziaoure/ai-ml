import joblib
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']

def main():
    model = joblib.load('models/text_clf.joblib')

    ds = load_dataset('ag_news')
    X_test = ds['test']['text']
    y_test = ds['test']['label']

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average='macro')

    print('=== Test Set Results (AG Nes officual test split) ===')
    print(f'Accuracy: {acc:.4f}')
    print(f'Macro F1: {macro_f1:.4f}\n')

    print(classification_report(y_test, preds, target_names=LABELS))
    print(f'Confusion Matrix:\n {confusion_matrix(y_test, preds)}')


if __name__ == '__main__':
    main()