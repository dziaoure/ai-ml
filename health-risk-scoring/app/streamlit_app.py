import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

import matplotlib.pyplot as plt
from pathlib import Path


#-----------------------------------------------
#   1. App Configuration
# ----------------------------------------------
st.set_page_config(
    page_title = 'Health Risk Scoring Demo',
    page_icon = '❤️',
    layout = 'centered'
)


#-----------------------------------------------
#   1. App Styles
# ----------------------------------------------
def load_css():
    css_path = Path(__file__).resolve().parent / "styles.css"

    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

load_css()

st.title('Health Risk Scoring Model')
st.markdown(
    '''
    This app uses a logistic regression model trained on a Framingham-style dataset 
    to estimate a patient's **10-year coronary heart disease (CHD) risk**.

    > ⚠️ **Disclaimer:** This tool is for **demonstration and educational purposes only**  
    > and is **not** intended for clinical use, diagnosis, or medical decision-making.
    '''
)

#-----------------------------------------------
#   2. Load trained model pipeline
# ----------------------------------------------

def load_artifacts():
    model_path = Path(__file__).resolve().parent.parent / 'models' / 'health_risk_model.joblib'
    data_path = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'framingham_heart_study.csv'

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    print(f'DF: {df.head()}')
    X = df.drop(columns = ['TenYearCHD'])

    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']

    if len(X) > 800:
        X_bg_sample = X.sample(800, random_state = 42)
    else:
        X_bg_sample = X

    X_bg_proc = preprocessor.transform(X_bg_sample)

    feature_names = X.columns.tolist()
    explainer = shap.LinearExplainer(classifier, X_bg_proc, feature_names = feature_names)

    return model, preprocessor, classifier, feature_names, explainer

model, preprocessor, classifier, feature_names, explainer = load_artifacts()


#-----------------------------------------------
#   3. Risk-tier helper
# ----------------------------------------------
def risk_tier(prob: float) -> str:
    if prob < 0.20:
        return 'Low'
    elif prob < 0.50:
        return 'Medium'
    else:
        return 'High'
    
def styled_risk_info(label, value, tier):
    colors = {
        "Low": "#FFFFFF",
        "Medium": "#ff9f0a",
        "High": "#ff453a"
    }

    color = colors.get(tier, "#FFFFFF")

    html = f"""
    <div class="stMetric" data-testid="stMetric">
        <p style="margin: 0; padding: 0;">{label}</p>
        <div data-testid="stMetricValue">
            <div style="color: {color}; line-height: normal;">
                {value}
            </div>
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


#-----------------------------------------------
#   4. Input form
# ----------------------------------------------
st.markdown('<div class="main">', unsafe_allow_html = True)

st.subheader('Enter Patient Information')
st.markdown(
    '''
    The inputs below represent a simplified subset of features from the Framingham Heart Study.
    Adjust them to explore how different factors affect the predicted 10-year CHD risk.
    '''
)

with st.form('risk_form'):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age (years)', min_value = 20, max_value = 90, value = 30)
        male = st.selectbox('Sex', options = ['Female', 'Male'])
        current_smoker = st.selectbox(
            'Current Smoker',
            options = ['No', 'Yes']
        )
        cigs_per_day = st.number_input('Cigarettes per day', min_value = 0, max_value = 60, value = 0)
        sys_bp = st.number_input('Systolic BP (mm/Hg)', min_value = 80, max_value = 260, value = 130)

    with col2:
        tot_chol = st.number_input('Total Cholesterol (mg/dL)', min_value = 100, max_value = 600, value = 220)
        glucose = st.number_input('Glucose (mg/dL)', min_value = 50, max_value = 300, value = 90)
        bmi = st.number_input('BMI', min_value = 10.0, max_value = 60.0, value = 26.0, step = 0.1)
        heart_rate = st.number_input('Heart Rate (bpm)', min_value = 30, max_value = 200, value = 75)
        dia_bp = st.number_input('Diastolicc BP (mm/Hg)', min_value = 40, max_value = 200, value = 40)

    st.markdown('### Additional Clinical Flags')

    col3, col4 = st.columns(2)

    with col3:
        prevalent_hyp = st.selectbox(
            'Prevalent Hypertension (diagnosed)',
            options = ['No', 'Yes']
        )
        bp_meds = st.selectbox(
            'On BP Medication',
            options = ['No', 'Yes']
        )
        prevalent_stroke = st.selectbox(
            'Prevalent Stroke (diagnosed)',
            options = ['No', 'Yes']
        )

    with col4:
        diabetes = st.selectbox(
            'Diabetes',
            options = ['No', 'Yes']
        )
        education = st.selectbox(
            'Education Level (Framingham coding)',
            options = ['1 (Some HS)', '2 (HS Grad)', '3 (Some College)', '4 (College Grad)']
        )

    submitted = st.form_submit_button('Calculate Risk')


#-----------------------------------------------
#   5. Prepare model input and run prediction
# ----------------------------------------------

if submitted:
    # Map categorical selections to numeric as in the training data
    male_num = 1 if male == 'Male' else 0
    current_smoker_num = 1 if current_smoker == 'Yes' else 0
    prevalent_hyp_num = 1 if prevalent_hyp == 'Yes' else 0
    bp_meds_num = 1 if bp_meds == 'Yes' else 0
    diabetes_num = 1 if diabetes == 'Yes' else 0
    prevalent_stroke_num = 1 if prevalent_stroke == 'Yes' else 0

    # Framingham often codes education as 1-4
    edu_mapping = {
        '1 (Some HS)': 1, 
        '2 (HS Grad)': 2, 
        '3 (Some College)': 3, 
        '4 (College Grad)': 4
    }
    education_num = edu_mapping[education]

    # IMPORTANT
    # Make sure these feature names and order match the training data.
    # Adjust as needed to reflect the exact columns used in the model
    input_dict = {
        'age': age,
        'male': male_num,
        'currentSmoker': current_smoker_num,
        'cigsPerDay': cigs_per_day,
        'sysBP': sys_bp,
        'diaBP': dia_bp,
        'totChol': tot_chol,
        'glucose': glucose,
        'BMI': bmi,
        'heartRate': heart_rate,
        'prevalentHyp': prevalent_hyp_num,
        'prevalentStroke': prevalent_stroke_num,
        'BPMeds': bp_meds_num,
        'diabetes': diabetes_num,
        'education': education_num
    }

    # currentSmoker (0, 1)
    # prevalentStroke (0, 1)
    # diaBP
    input_df = pd.DataFrame([input_dict])
    
    #-----------------------------------------------
    #   6. Run the model
    # ----------------------------------------------
    prob = model.predict_proba(input_df)[0, 1]
    tier = risk_tier(prob)


    #-----------------------------------------------
    #   7. Display the results
    # ----------------------------------------------

    st.subheader('Predicted 10-Year CHD Risk')

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        styled_risk_info(label = 'Estimated Risk Probability', value = f'{prob:.1%}', tier = tier)

    with col_res2:
        styled_risk_info(label = 'Risk Tier', value = tier, tier = tier)


    st.markdown('---')
    st.markdown(
        '''
        ### Interpretation

        - The **estimated probability** represents this patient's modeled risk of developing
        coronary heart disease within 10 years, based on the features provided.
        - The **risk tier** is derived from simple thresholds:
        - **Low:** p < 0.20  
        - **Medium:** 0.20 ≤ p < 0.50  
        - **High:** p ≥ 0.50  

        Remember, this app is a **demo** built from a historical research dataset and is **not**
        intended for real clinical decision-making.
    '''
    )


    #-----------------------------------------------
    #   8. SHAP explanation for this patient
    # ----------------------------------------------
    st.subheader('Why did the model predict this risk?')
    st.markdown(
        '''
        The plot below shows how each feature contributed to this patient's predicted risk.
        Features pushing the risk **higher** appear in one direction, while those **lowering** risk
        appear in the opposite direction.
        '''
    )

    # Transform the single input using the same preprocessor
    X_input_proc = preprocessor.transform(input_df)

    # COmpute the SHAP values for this patient
    numeric_columns = preprocessor.transformers_[0][2]
    
    shap_values = explainer(X_input_proc)

    # Waterfall plot fo the sample
    # Note: `shap.plots.waterfall` returns a `Matplotlib Figure` when `show = False`
    shap.plots.waterfall(shap_values[0], max_display = 8, show = False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.clf()

    st.markdown(
        '''
        > **Note:** This explanation is based on a logistic regression model trained on
        > a historical research dataset. It is meant to illustrate model interpretability,
        > not to provide clinical guidance.
        '''
    )

    st.markdown("---")
    st.markdown(
        '''
        ### Interpretation

        - Positive contributions (bars extending to the right) indicate features that **increase** the predicted CHD risk for this patient.
        - Negative contributions (bars to the left) indicate features that **reduce** the risk.
        - You can experiment with the input values above to see how changes in age, smoking, blood pressure, cholesterol, and glucose affect the model's reasoning.
        '''
    )

st.markdown('</div>', unsafe_allow_html = True)