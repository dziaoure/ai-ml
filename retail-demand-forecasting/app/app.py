from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import pandas as pd
import joblib

# --- Fix imports when runing `streamlit` from `/app`
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


#-----------------------------------------------
#           Load artifacts
#-----------------------------------------------
model_path = Path(__file__).resolve().parent.parent / 'models/gradient_boosting_model.joblib'
model = joblib.load(model_path)

feature_columns_path = Path(__file__).resolve().parent.parent / 'models/feature_columns.joblib'
feature_columns = joblib.load(feature_columns_path)

csv_file_path = Path(__file__).resolve().parent.parent / 'data/processed/model_ready_data.csv'
df = pd.read_csv(csv_file_path, parse_dates=['Date'])


#-----------------------------------------------
#           Streamlist UI
#-----------------------------------------------
st.set_page_config(page_title = 'Retail Demand Forecasting - Walmart Weekly Sales', layout = 'wide')

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    image_path = Path(__file__).resolve().parent.parent / 'images/walmart-spark.png'
    st.image(image_path, width=100)

st.title('Retail Demand Forecasting - Walmart')
st.write('')

#-----------------------------------------------
#   App Styles
# ----------------------------------------------
def load_css():
    css_path = Path(__file__).resolve().parent / "styles.css"

    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)

load_css()

# Sidebar controls
store_id = st.sidebar.selectbox('Select Store', sorted(df['Store'].unique()))

store_df = df[df['Store'] == store_id].sort_values('Date')
row = store_df.iloc[[-1]]

X_inference = row[feature_columns]  # Only featuures seen at fit time

# Predictons
prediction = float(model.predict(X_inference)[0])

# Comparison metrics
last_actual = float(row['Weekly_Sales'].iloc[0])

delta = prediction - last_actual
pct_Change = (delta / last_actual * 100) if last_actual != 0 else None

st.write(f'Store ID: {store_id}')
col1, col2, col3 = st.columns(3)
col1.metric('Predicted Weekly Sales', f'${prediction:,.0f}')
col2.metric('Last Actual Weekly Sales', f'${last_actual:,.0f}')

if pct_Change is None:
    col3.metric('Prediction vs Last Week', f'${delta:,.0f}')
else:
    col3.metric('Prediction vs Last Week', f'${delta:,.0f}', f'{pct_Change:,.1f}%')

pred_date = row['Date'].iloc[0]
st.caption(f'Prediction date: {pred_date.date()}')

st.caption(
    "Context: This prediction is primarily driven by recent demand history "
    "(last week’s sales and 4–8 week rolling averages), with smaller adjustments "
    "from seasonality (week-of-year), holidays, and macro variables."
)

#  Add a "Recent Sales Trend" chart
st.subheader('Recent Sales Trend (Last 20 Weeks)')

trend = store_df[['Date', 'Weekly_Sales']].tail(20).copy()
trend['Prediction'] = pd.NA
trend.loc[trend.index[-1], 'Prediction'] = prediction   # Show prediction on last date

trend = trend.set_index('Date')
st.line_chart(trend)
