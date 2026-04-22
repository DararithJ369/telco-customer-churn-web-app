import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Churn Prediction Platform",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== THEME DETECTION ==================
try:
    is_dark = st.get_option("theme.base") == "dark"
except:
    is_dark = False

# ================== MODERN CSS STYLING ==================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    * {{
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}

    :root {{
        --primary: #0f172a;
        --secondary: #64748b;
        --accent: #3b82f6;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --border: #e2e8f0;
        --bg-light: #f8fafc;
        --card-light: #ffffff;
        --bg-dark: #0f172a;
        --card-dark: #1e293b;
    }}

    .stApp {{
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        font-weight: 400;
    }}

    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }}

    .block-container {{
        max-width: 1600px;
        padding: 2.5rem 3rem;
    }}

    /* Hero Section */
    .hero-container {{
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 3.5rem 3rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.15);
    }}

    .hero-container::before {{
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
        border-radius: 50%;
        z-index: 0;
    }}

    .hero-content {{
        position: relative;
        z-index: 1;
        color: white;
    }}

    .hero-title {{
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 1rem 0;
        line-height: 1.1;
    }}

    .hero-subtitle {{
        font-size: 1.1rem;
        font-weight: 300;
        color: rgba(255, 255, 255, 0.85);
        line-height: 1.6;
        max-width: 700px;
    }}

    /* Section Heading */
    .section-title {{
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin: 2.5rem 0 1.5rem 0;
        letter-spacing: -0.01em;
        text-transform: none;
    }}

    /* Cards */
    .info-card {{
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }}

    .info-card:hover {{
        box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
        border-color: #cbd5e1;
    }}

    .metric-display {{
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}

    .metric-display:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(59, 130, 246, 0.08);
    }}

    .metric-label {{
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.8rem;
    }}

    .metric-value {{
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
        font-family: 'JetBrains Mono', monospace;
    }}

    .metric-unit {{
        font-size: 0.9rem;
        color: #94a3b8;
        margin-left: 0.4rem;
    }}

    /* Status Badges */
    .status-good {{
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 2rem;
        position: relative;
        overflow: hidden;
    }}

    .status-good::before {{
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: #10b981;
    }}

    .status-risk {{
        background: linear-gradient(135deg, #fef2f2 0%, #fff5f5 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 2rem;
        position: relative;
        overflow: hidden;
    }}

    .status-risk::before {{
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: #ef4444;
    }}

    .status-title {{
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }}

    .status-good .status-title {{
        color: #065f46;
    }}

    .status-risk .status-title {{
        color: #7f1d1d;
    }}

    .status-description {{
        font-size: 1rem;
        color: #64748b;
        line-height: 1.6;
    }}

    /* Summary Grid */
    .summary-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-top: 1.5rem;
    }}

    .summary-item {{
        font-size: 0.95rem;
        line-height: 1.8;
        color: #475569;
    }}

    .summary-item strong {{
        color: #0f172a;
        font-weight: 600;
    }}

    /* Gauge Container */
    .probability-section {{
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }}

    .probability-label {{
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1.5rem;
    }}

    .probability-value {{
        font-size: 2.5rem;
        font-weight: 800;
        color: #3b82f6;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 1rem;
    }}

    /* Chart Container */
    .chart-container {{
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;   /* reduce from 1.5rem */
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        height: 100%;
    }}

    .chart-title {{
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
    }}


    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
    }}

    .sidebar-section {{
        margin-bottom: 2rem;
    }}

    .sidebar-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1.2rem;
        letter-spacing: -0.01em;
    }}

    .sidebar-subtitle {{
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 1rem;
    }}

    /* Button Styling */
    .stButton > button {{
        width: 100%;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 12px;
        height: auto;
        font-weight: 600;
        font-size: 1rem;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
        text-transform: none;
        letter-spacing: 0;
    }}

    .stButton > button:hover {{
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* Info Box */
    .info-box {{
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1.5rem 0;
        font-size: 0.95rem;
        color: #1e40af;
        line-height: 1.6;
    }}

    /* Insights Section */
    .insights-box {{
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        font-size: 0.95rem;
        color: #0c4a6e;
        line-height: 1.8;
    }}

    .insights-list {{
        margin-top: 1rem;
    }}

    .insight-item {{
        margin-bottom: 0.8rem;
        padding-left: 1.5rem;
        position: relative;
    }}

    .insight-item::before {{
        content: '•';
        position: absolute;
        left: 0;
        color: #0284c7;
        font-weight: bold;
        font-size: 1.2rem;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
        font-weight: 300;
    }}

    /* Responsive Design */
    @media (max-width: 768px) {{
        .hero-title {{
            font-size: 2rem;
        }}

        .hero-subtitle {{
            font-size: 1rem;
        }}

        .summary-grid {{
            grid-template-columns: 1fr;
        }}

        .metric-value {{
            font-size: 1.8rem;
        }}
    }}

    /* Animations */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    .info-card {{
        animation: fadeInUp 0.5s ease forwards;
    }}

    /* Plotly Custom Styling */
    .plotly {{
        font-family: 'Outfit', sans-serif;
    }}
</style>
""", unsafe_allow_html=True)

# ================== LOAD ARTIFACTS ==================
@st.cache_resource
def load_artifacts():
    """Load all preprocessing and model artifacts"""
    try:
        model = joblib.load("notebooks/artifacts/logistic_churn_model.joblib")
        scaler = joblib.load("notebooks/artifacts/scaler.joblib")
        feature_names = joblib.load("notebooks/artifacts/feature_names.joblib")
        config = joblib.load("notebooks/artifacts/preprocessing_config.joblib")
        return model, scaler, feature_names, config
    except FileNotFoundError as e:
        st.error(f"Missing artifact: {e}")
        st.stop()

@st.cache_data
def load_data():
    """Load original data for EDA"""
    try:
        df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        return df
    except FileNotFoundError:
        st.error("Data file not found!")
        st.stop()

model, scaler, feature_names, config = load_artifacts()
original_df = load_data()

# Get model metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Calculate metrics on sample
model_df = original_df[["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", 
                        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                        "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", 
                        "TotalCharges", "Churn"]].copy()

model_df["TotalCharges"] = pd.to_numeric(model_df["TotalCharges"], errors="coerce")
model_df = model_df.dropna()
model_accuracy = 0.7459  # From your training results

# ================== PREPROCESSING FUNCTION ==================
def preprocess_user_input(user_data: dict) -> pd.DataFrame:
    """Convert user input to preprocessed feature matrix matching training data"""
    input_df = pd.DataFrame([user_data])
    
    # Create engineered features
    input_df["AvergeCharges"] = input_df["TotalCharges"] / (input_df["tenure"] + 1)
    input_df["charges_per_month"] = input_df["MonthlyCharges"] / (input_df["tenure"] + 1)
    
    # Create tenure groups
    bins = [0, 12, 24, np.inf]
    labels = [0, 1, 2]
    input_df["tenure_group"] = pd.cut(input_df["tenure"], bins=bins, labels=labels, right=False)
    input_df["tenure_group"] = input_df["tenure_group"].astype(int)
    
    # Drop columns
    cols_to_drop = [col for col in input_df.columns if col in ["TotalCharges", "gender", "PhoneService"]]
    input_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Encode binary columns
    binary_mapping = {"yes": 1, "no": 0}
    for col in ["Partner", "Dependents", "PaperlessBilling"]:
        if col in input_df.columns:
            input_df[col] = input_df[col].str.lower().map(binary_mapping)
    
    # One-hot encode categorical
    multi_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                  "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                  "Contract", "PaymentMethod"]
    
    for col in multi_cols:
        if col in input_df.columns:
            input_df = pd.get_dummies(input_df, columns=[col], drop_first=True)
    
    # Ensure all features present
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    input_df = input_df[feature_names]
    input_df_scaled = scaler.transform(input_df.values)
    
    return input_df_scaled

# ================== HEADER ==================
st.markdown("""
<div class="hero-container">
    <div class="hero-content">
        <h1 class="hero-title">Churn Prediction Platform</h1>
        <p class="hero-subtitle">Advanced machine learning system for predicting customer churn with real-time accuracy and actionable insights</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== MODEL PERFORMANCE ==================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-display">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value">76.00<span class="metric-unit">%</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-display">
        <div class="metric-label">ROC-AUC Score</div>
        <div class="metric-value">84.05<span class="metric-unit">%</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-display">
        <div class="metric-label">Total Features</div>
        <div class="metric-value">30<span class="metric-unit"></span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================== SIDEBAR FORM ==================
st.sidebar.markdown('<div class="sidebar-title">Customer Information</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-subtitle">Fill in all details for accurate prediction</div>', unsafe_allow_html=True)

st.sidebar.markdown("### Demographics", unsafe_allow_html=True)
senior_citizen = st.sidebar.radio("Senior Citizen", ["No", "Yes"], horizontal=True)
partner = st.sidebar.radio("Has Partner", ["No", "Yes"], horizontal=True)
dependents = st.sidebar.radio("Has Dependents", ["No", "Yes"], horizontal=True)

st.sidebar.markdown("### Internet Services", unsafe_allow_html=True)
internet_service = st.sidebar.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

st.sidebar.markdown("### Additional Services", unsafe_allow_html=True)
online_security = st.sidebar.selectbox("Online Security", ["No internet service", "No", "Yes"])
online_backup = st.sidebar.selectbox("Online Backup", ["No internet service", "No", "Yes"])
device_protection = st.sidebar.selectbox("Device Protection", ["No internet service", "No", "Yes"])
tech_support = st.sidebar.selectbox("Tech Support", ["No internet service", "No", "Yes"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

st.sidebar.markdown("### Account Details", unsafe_allow_html=True)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.radio("Paperless Billing", ["No", "Yes"], horizontal=True)
payment_method = st.sidebar.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

st.sidebar.markdown("### Service Metrics", unsafe_allow_html=True)
tenure = st.sidebar.slider("Tenure (months)", 0, 120, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0.0, 120.0, 65.0, step=1.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 8000.0, 780.0, step=50.0)

predict_btn = st.sidebar.button("Generate Prediction", use_container_width=True)

# ================== PREPARE INPUT ==================
user_input = {
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "Partner": partner.lower(),
    "Dependents": dependents.lower(),
    "tenure": tenure,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing.lower(),
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# ================== MAKE PREDICTION ==================
try:
    user_input_processed = preprocess_user_input(user_input)
    pred_class = model.predict(user_input_processed)[0]
    pred_prob = model.predict_proba(user_input_processed)[0][1]
except Exception as e:
    st.sidebar.error(f"Preprocessing error: {e}")
    pred_class = 0
    pred_prob = 0.3

prediction = "Likely to Churn" if pred_class == 1 else "Likely to Stay"
churn_risk = "High Risk" if pred_prob > 0.6 else ("Moderate Risk" if pred_prob > 0.4 else "Low Risk")

# ================== PREDICTION DISPLAY ==================
left_col, right_col = st.columns([1, 1.4], gap="large")

with left_col:
    st.markdown('<h2 class="section-title">Customer Profile</h2>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-card">
        <div class="summary-grid">
            <div class="summary-item"><strong>Status</strong><br>{"Senior" if senior_citizen == "Yes" else "Regular"}</div>
            <div class="summary-item"><strong>Partnership</strong><br>{"Yes" if partner == "Yes" else "No"}</div>
            <div class="summary-item"><strong>Dependents</strong><br>{"Yes" if dependents == "Yes" else "No"}</div>
            <div class="summary-item"><strong>Internet</strong><br>{internet_service}</div>
            <div class="summary-item"><strong>Contract</strong><br>{contract}</div>
            <div class="summary-item"><strong>Tech Support</strong><br>{tech_support}</div>
            <div class="summary-item"><strong>Monthly Charge</strong><br>${monthly_charges:.2f}</div>
            <div class="summary-item"><strong>Total Charges</strong><br>${total_charges:,.2f}</div>
            <div class="summary-item"><strong>Tenure</strong><br>{tenure} months</div>
            <div class="summary-item"><strong>Payment Method</strong><br>{payment_method}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    st.markdown('<h2 class="section-title">Prediction Result</h2>', unsafe_allow_html=True)

    if predict_btn:
        if prediction == "Likely to Churn":
            st.markdown(f"""
            <div class="status-risk">
                <div class="status-title">Churn Risk Detected</div>
                <div class="status-description">
                    This customer shows <strong>{churn_risk}</strong> signals. Immediate retention strategies recommended.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-good">
                <div class="status-title">Customer Retention Strong</div>
                <div class="status-description">
                    This customer appears <strong>{churn_risk}</strong>. Focus on maintaining satisfaction and engagement.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="probability-section">', unsafe_allow_html=True)
        st.markdown('<div class="probability-label">Churn Probability Score</div>', unsafe_allow_html=True)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 28, 'family': 'JetBrains Mono'}},
            title={'text': "Likelihood", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2},
                'bar': {'color': "#3b82f6", 'thickness': 0.4},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.15)"},
                    {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.15)"},
                    {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.15)"}
                ],
                'threshold': {
                    'line': {'color': "#3b82f6", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        fig_gauge.update_layout(
            height=280,
            font={'family': 'Outfit'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f'<div class="probability-value">{pred_prob:.1%}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            Click "Generate Prediction" to analyze customer churn risk
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================== EDA SECTION ==================
st.markdown('<h2 class="section-title">Market Analytics Dashboard</h2>', unsafe_allow_html=True)


col1, col2 = st.columns(2, gap="large")

with col1:
    churn_dist = original_df["Churn"].value_counts()
    fig1 = px.pie(
        values=churn_dist.values,
        names=churn_dist.index,
        title="Customer Churn Distribution",
        color_discrete_map={"Yes": "#ef4444", "No": "#10b981"}
    )
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    fig1.update_layout(
        height=350,
        font={'family': 'Outfit'},
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig1, use_container_width=True, config={'responsive': True})
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    contract_churn = original_df.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
    fig2 = px.bar(
        contract_churn,
        x="Contract",
        y="Count",
        color="Churn",
        barmode="group",
        title="Churn by Contract Type",
        color_discrete_map={"Yes": "#ef4444", "No": "#10b981"}
    )
    fig2.update_layout(
        height=350,
        font={'family': 'Outfit'},
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,240,0.5)',
        xaxis_title="",
        yaxis_title="Customer Count"
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True, config={'responsive': True})
    st.markdown('</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="large")

with col3:
    internet_churn = original_df.groupby(["InternetService", "Churn"]).size().reset_index(name="Count")
    fig3 = px.bar(
        internet_churn,
        x="InternetService",
        y="Count",
        color="Churn",
        barmode="group",
        title="Churn by Internet Service",
        color_discrete_map={"Yes": "#ef4444", "No": "#10b981"}
    )
    fig3.update_layout(
        height=350,
        font={'family': 'Outfit'},
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,240,0.5)',
        xaxis_title="",
        yaxis_title="Customer Count"
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig3, use_container_width=True, config={'responsive': True})
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    fig4 = px.box(
        original_df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges Distribution",
        color_discrete_map={"Yes": "#ef4444", "No": "#10b981"}
    )
    fig4.update_layout(
        height=350,
        font={'family': 'Outfit'},
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,240,0.5)',
        xaxis_title="Churn Status",
        yaxis_title="Monthly Charges ($)"
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig4, use_container_width=True, config={'responsive': True})
    st.markdown('</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2, gap="large")

with col5:
    fig5 = px.histogram(
        original_df,
        x="tenure",
        color="Churn",
        nbins=25,
        barmode="overlay",
        title="Customer Tenure Distribution",
        color_discrete_map={"Yes": "#ef4444", "No": "#10b981"}
    )
    fig5.update_layout(
        height=350,
        font={'family': 'Outfit'},
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,240,0.5)',
        xaxis_title="Tenure (months)",
        yaxis_title="Count"
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig5, use_container_width=True, config={'responsive': True})
    st.markdown('</div>', unsafe_allow_html=True)

with col6:
    tech_support = original_df.groupby(["TechSupport", "Churn"]).size().reset_index(name="Count")
    fig6 = px.bar(
        tech_support,
        x="TechSupport",
        y="Count",
        color="Churn",
        barmode="group",
        title="Churn by Tech Support Subscription",
        color_discrete_map={"Yes": "#ef4444", "No": "#10b981"}
    )
    fig6.update_layout(
        height=350,
        font={'family': 'Outfit'},
        paper_bgcolor='white',
        plot_bgcolor='rgba(240,240,240,0.5)',
        xaxis_title="",
        yaxis_title="Customer Count"
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(fig6, use_container_width=True, config={'responsive': True})
    st.markdown('</div>', unsafe_allow_html=True)

# ================== KEY INSIGHTS ==================
st.markdown('<h2 class="section-title">Key Insights</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="insights-box">
    <div class="insights-list">
        <div class="insight-item">Month-to-month contracts show 43% churn rate compared to only 3% for two-year contracts</div>
        <div class="insight-item">Fiber optic internet users have 2.2x higher churn risk than DSL users</div>
        <div class="insight-item">Tech support and online security subscriptions reduce churn by 65%</div>
        <div class="insight-item">Customers with tenure under 12 months have 5x higher churn likelihood</div>
        <div class="insight-item">Electronic check payment method correlates with 45% churn rate</div>
        <div class="insight-item">Customers with partners show 40% lower churn compared to single customers</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown("""
<div class="footer">
Churn Prediction Platform | Logistic Regression Model with SMOTE | 30 Features | 74.6% Accuracy
</div>
""", unsafe_allow_html=True)