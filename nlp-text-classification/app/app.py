from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path

LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']

def project_root() -> Path:
    # app/streamlist_app.py -> `app/` -> project root
    return Path(__file__).resolve().parents[1]


@st.cache_resource
def load_model():
    model_path = project_root() / 'models' / 'text_clf.joblib'

    if not model_path.exists():
        raise FileNotFoundError(
            f'Model not found a {model_path}. Run `python src/tune.py` first'
        )
    
    return joblib.load(model_path)


def softmax(x: np.ndarray) -> np.ndarray:
    # Stable softmax for nice "confidence-like" display (not true probabilities)
    x = x - np.max(x)
    exps = np.exp(x)
    return exps / np.sum(exps)


def explain_prediction(pipeline, text: str, top_k: int = 15) -> pd.DataFrame:
    '''
    Returns top-k token contributions for the predicted class.
    Contribution = tfidf(token) * weight(token, class)
    '''
    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']

    X = tfidf.transform([text]) # sparse row vector

    # Predicted class index
    pred_idx = int(pipeline.predict([text])[0])

    feature_names = np.array(tfidf.get_feature_names_out())
    class_weights = clf.coef_[pred_idx]

    # Controbutions for tokens present in this document
    # X is sparse; only nonzero `tdidf` matter
    row = X.tocoo()
    tokens = feature_names[row.col]
    tfidf_vals = row.data
    weights = class_weights[row.col]
    contrib = tfidf_vals * weights
    df = pd.DataFrame({
        'tokens': tokens,
        'tfidf': tfidf_vals,
        'weight': weights,
        'contribution': contrib    
    })

    # Sort by contribution (largest positive pushes toward predicted class)
    df = df.sort_values(by = 'contribution', ascending=False).head(top_k).reset_index(drop=True)
    return df

#-----------------------------------------------
#   App Styles
# ----------------------------------------------
def load_css():
    css_path = Path(__file__).resolve().parent / "styles.css"

    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)


def main():
    st.set_page_config(page_title='AG News Classifier', layout='wide')

    load_css()

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        image_path = project_root() / 'images/news-icon.png'
        st.image(image_path, width=120)

    st.title('AG News Classifier')
    subtitle_html = """
        <p style="text-align: center;color:#fff; opacity:0.65; margin-top: 0;">
        Classifies text into <strong>World</strong>, <strong>Sports</strong>, <strong>Business</strong>, or 
        <strong>Sci/Tech</strong> using a tuned <strong>TF-IDF + Linear SVM</strong> pipeline.
        </p>
    """
    st.markdown(subtitle_html, unsafe_allow_html=True)

    model = load_model()

    with st.sidebar:
        st.header('Settings')
        top_k = st.slider('Top contributing tokens', min_value=5, max_value=30, value=15, step=1)

        st.markdown('---')
        st.caption('Note: LinearSVC outouts decision scores, not true probabilities.')

    example_world = "The government announced new talks amid rising tensions in the region."
    example_sports = "The coach said the team is ready for the cup final after a strong season."
    example_business = "Oil prices rose as the bank raised rates and markets reacted to the tax plan."
    example_scitech = "NASA scientists released new findings about space telescopes and software updates."

    st.markdown('---')
    
    colA, colB, colC, colD = st.columns(4)

    with colA:
        if st.button('World example'):
            st.session_state['text_input'] = example_world

    with colB:
        if st.button('Sports example'):
            st.session_state['text_input'] = example_sports

    with colC:
        if st.button('Business example'):
            st.session_state['text_input'] = example_business

    with colD:
        if st.button('Sci/Tech example'):
            st.session_state['text_input'] = example_scitech

    default_text = st.session_state.get('text_input', '')
    text = st.text_area('Paste a news article / snippet', value=default_text, height=220)

    if st.button('Classify', type='primary', disabled=(len(text.strip()) == 0)):
        pred_idx = int(model.predict([text])[0])
        pred_label = LABELS[pred_idx]

        # DEcision scores (one per class)
        scores = model.decision_function([text])[0]
        scores = np.array(scores, dtype=float)

        # Confidence-like display (NOT a real probability)
        conf_like = softmax(scores)

        st.subheader('Prediction')
        st.metric('Predicted topic', pred_label)

        score_df = pd.DataFrame({
            'Class': LABELS,
            'Decision score': scores,
            'Softmax(score) (Confidence-like)': conf_like
        }).sort_values('Decision score', ascending=False)

        st.write('### Class scores')
        st.dataframe(score_df, width='stretch')

        st.write('### Top token contributions (for predicted class)')
        expl = explain_prediction(model, text, top_k=top_k)

        # Marke it extra readable
        expl_display = expl.copy()
        expl_display['tfidf'] = expl_display['tfidf'].map(lambda x: float(f'{x:.4f}'))
        expl_display['weight'] = expl_display['weight'].map(lambda x: float(f'{x:.4f}'))
        expl_display['contribution'] = expl_display['contribution'].map(lambda x: float(f'{x:.4f}'))

        st.dataframe(expl_display, width='stretch')

        st.caption(
            'Contribution = TF-IDF(token) * weight(token, predicted_class). '
            'Higher positive contributions push the prediction toward the selected class.'
        )


if __name__ == '__main__':
    main()
