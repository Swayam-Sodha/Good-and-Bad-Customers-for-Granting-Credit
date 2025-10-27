import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import random

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Credit Card Default Dashboard", layout="wide")
st.title("💳 Credit Card Default Prediction & Dashboard")

# --------------------------
# Load Dataset
# --------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("Dataset Loaded!")
else:
    default_csv_path = "Credit_Card_Default.csv"
    df = load_data(default_csv_path)
    st.info("Using default CSV dataset")

# --------------------------
# Load Model
# --------------------------
try:
    model = pickle.load(open("credit_default_model.pkl", "rb"))
except Exception:
    model = None
    st.warning("Model not found. Predictions will not work.")

# --------------------------
# Tabs for UI
# --------------------------
tab1, tab2, tab3 = st.tabs(["📊 EDA", "💻 Predict Default", "🎲 Random Example Generator"])

# --------------------------
# EDA Tab
# --------------------------
with tab1:
    st.subheader("Exploratory Data Analysis")
    with st.expander("Target Distribution"):
        if 'default.payment.next.month' in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(x='default.payment.next.month', data=df, ax=ax)
            st.pyplot(fig)

    with st.expander("Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(12,8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    with st.expander("Numeric Feature Distributions"):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

# --------------------------
# Sidebar Inputs (persistent)
# --------------------------
st.sidebar.header("Customer Information")

random_example = {
    'LIMIT_BAL': random.randint(10000, 1000000),
    'SEX': random.choice([1,2]),
    'EDUCATION': random.choice([1,2,3,4]),
    'MARRIAGE': random.choice([1,2,3]),
    'AGE': random.randint(18, 75),
    'PAY_0': random.randint(-2,8),
    'PAY_2': random.randint(-2,8),
    'PAY_3': random.randint(-2,8),
    'PAY_4': random.randint(-2,8),
    'PAY_5': random.randint(-2,8),
    'PAY_6': random.randint(-2,8),
    'BILL_AMT1': random.randint(0, 1000000),
    'BILL_AMT2': random.randint(0, 1000000),
    'BILL_AMT3': random.randint(0, 1000000),
    'BILL_AMT4': random.randint(0, 1000000),
    'BILL_AMT5': random.randint(0, 1000000),
    'BILL_AMT6': random.randint(0, 1000000),
    'PAY_AMT1': random.randint(0, 500000),
    'PAY_AMT2': random.randint(0, 500000),
    'PAY_AMT3': random.randint(0, 500000),
    'PAY_AMT4': random.randint(0, 500000),
    'PAY_AMT5': random.randint(0, 500000),
    'PAY_AMT6': random.randint(0, 500000)
}

def user_input_field(name, default=0, min_val=None, max_val=None):
    val = random_example.get(name, default)
    if min_val is not None:
        val = max(val, min_val)
    if max_val is not None:
        val = min(val, max_val)
    return st.sidebar.number_input(f"{name}", value=int(val), min_value=min_val, max_value=max_val)

LIMIT_BAL = user_input_field("Credit Limit (LIMIT_BAL)", min_val=0)
SEX = st.sidebar.selectbox("SEX (1=Male, 2=Female)", [1,2], index=random_example['SEX']-1)
EDUCATION = st.sidebar.selectbox("EDUCATION (1=Grad,2=Uni,3=High School,4=Others)", [1,2,3,4], index=random_example['EDUCATION']-1)
MARRIAGE = st.sidebar.selectbox("MARRIAGE (1=Married,2=Single,3=Others)", [1,2,3], index=random_example['MARRIAGE']-1)
AGE = user_input_field("AGE", min_val=18, max_val=100)

PAY_0 = user_input_field("PAY_0 (Sep)", min_val=-2, max_val=8)
PAY_2 = user_input_field("PAY_2 (Aug)", min_val=-2, max_val=8)
PAY_3 = user_input_field("PAY_3 (Jul)", min_val=-2, max_val=8)
PAY_4 = user_input_field("PAY_4 (Jun)", min_val=-2, max_val=8)
PAY_5 = user_input_field("PAY_5 (May)", min_val=-2, max_val=8)
PAY_6 = user_input_field("PAY_6 (Apr)", min_val=-2, max_val=8)

BILL_AMT1 = user_input_field("BILL_AMT1 (Sep)", min_val=0)
BILL_AMT2 = user_input_field("BILL_AMT2 (Aug)", min_val=0)
BILL_AMT3 = user_input_field("BILL_AMT3 (Jul)", min_val=0)
BILL_AMT4 = user_input_field("BILL_AMT4 (Jun)", min_val=0)
BILL_AMT5 = user_input_field("BILL_AMT5 (May)", min_val=0)
BILL_AMT6 = user_input_field("BILL_AMT6 (Apr)", min_val=0)

PAY_AMT1 = user_input_field("PAY_AMT1 (Sep)", min_val=0)
PAY_AMT2 = user_input_field("PAY_AMT2 (Aug)", min_val=0)
PAY_AMT3 = user_input_field("PAY_AMT3 (Jul)", min_val=0)
PAY_AMT4 = user_input_field("PAY_AMT4 (Jun)", min_val=0)
PAY_AMT5 = user_input_field("PAY_AMT5 (May)", min_val=0)
PAY_AMT6 = user_input_field("PAY_AMT6 (Apr)", min_val=0)

input_data = pd.DataFrame({
    'LIMIT_BAL':[LIMIT_BAL], 'SEX':[SEX], 'EDUCATION':[EDUCATION], 'MARRIAGE':[MARRIAGE], 'AGE':[AGE],
    'PAY_0':[PAY_0], 'PAY_2':[PAY_2], 'PAY_3':[PAY_3], 'PAY_4':[PAY_4], 'PAY_5':[PAY_5], 'PAY_6':[PAY_6],
    'BILL_AMT1':[BILL_AMT1], 'BILL_AMT2':[BILL_AMT2], 'BILL_AMT3':[BILL_AMT3], 'BILL_AMT4':[BILL_AMT4], 'BILL_AMT5':[BILL_AMT5], 'BILL_AMT6':[BILL_AMT6],
    'PAY_AMT1':[PAY_AMT1], 'PAY_AMT2':[PAY_AMT2], 'PAY_AMT3':[PAY_AMT3], 'PAY_AMT4':[PAY_AMT4], 'PAY_AMT5':[PAY_AMT5], 'PAY_AMT6':[PAY_AMT6]
})

# --------------------------
# Predict Tab
# --------------------------
with tab2:
    st.subheader("Predict Default for a Customer")
    if st.button("Predict Default"):
        if model is not None:
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
            result_df = pd.DataFrame({
                "Prediction":["Will NOT default" if prediction[0]==0 else "Will default"],
                "Probability_No":[prediction_proba[0][0]],
                "Probability_Yes":[prediction_proba[0][1]]
            })

            st.subheader("Prediction Result")
            st.write(f"Customer is: **{result_df['Prediction'][0]}**")

            # Show probabilities with color
            st.subheader("Prediction Probability")
            st.dataframe(result_df.style.background_gradient(cmap="coolwarm"))

            # Provide CSV download
            csv = result_df.to_csv(index=False).encode()
            st.download_button(
                label="Download Prediction as CSV",
                data=csv,
                file_name="prediction.csv",
                mime="text/csv"
            )
        else:
            st.error("Model not found. Cannot make prediction.")

# --------------------------
# Random Example Generator Tab
# --------------------------
with tab3:
    st.subheader("🎲 Generate Random Customer Example")
    if st.button("Generate Random Example"):
        for key in random_example:
            if key == 'LIMIT_BAL':
                random_example[key] = random.randint(10000, 1000000)
            elif key == 'SEX':
                random_example[key] = random.choice([1,2])
            elif key == 'EDUCATION':
                random_example[key] = random.choice([1,2,3,4])
            elif key == 'MARRIAGE':
                random_example[key] = random.choice([1,2,3])
            elif key == 'AGE':
                random_example[key] = random.randint(18, 75)
            elif key.startswith('PAY_'):
                random_example[key] = random.randint(-2,8)
            elif key.startswith('BILL_AMT'):
                random_example[key] = random.randint(0, 1000000)
            elif key.startswith('PAY_AMT'):
                random_example[key] = random.randint(0, 500000)
        st.experimental_rerun()
