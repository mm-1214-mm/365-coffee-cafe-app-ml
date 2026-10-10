import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING (Modern Classic Dark Theme)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="365 Coffee Café | Customer Retention Intelligence",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global Dark Theme & Coffee Palette */
    .stApp {
        background-color: #121214;
        color: #E4E4E7;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #18181B;
        border-right: 1px solid #27272A;
    }
    
    /* Headers & Text */
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA !important;
        font-family: 'Playfair Display', serif;
    }
    
    p, span, label, div {
        color: #D4D4D8;
    }
    
    /* Metric Cards / Containers */
    .metric-card {
        background: linear-gradient(135deg, #1E1E24 0%, #18181B 100%);
        border: 1px solid #3F3F46;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        text-align: center;
        margin-bottom: 16px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #D97706;
    }
    
    .metric-label {
        font-size: 14px;
        color: #A1A1AA;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #D97706 0%, #B45309 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(217, 119, 6, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        box-shadow: 0 6px 16px rgba(217, 119, 6, 0.5);
    }
    
    /* Input Fields */
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: #1E1E24;
        border-radius: 8px;
    }
    
    /* Disclaimer Box */
    .disclaimer-box {
        background-color: #1E1E24;
        border-left: 4px solid #D97706;
        padding: 14px;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
        color: #A1A1AA;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODEL TRAINING CACHE (Simulating Backend Machine Learning Pipeline)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    # Load dataset to train our lightweight Random Forest classifier dynamically
    df = pd.read_excel('365 Coffee Cafe Dataset.xlsx', sheet_name='Dataset')
    
    # Aggregate to visit level (basket_id)
    visit_df = df.groupby('basket_id').agg({
        'store_location': 'first',
        'customer_age': 'first',
        'loyalty_tier': 'first',
        'visit_number': 'first',
        'days_since_previous_visit': 'first',
        'order_channel': 'first',
        'payment_method': 'first',
        'revenue': 'sum',
        'discount_pct': 'mean',
        'wait_time_min': 'mean',
        'service_rating': 'mean',
        'returned_within_14_days': 'first'
    }).reset_index()
    
    # Encode categorical features
    encoders = {}
    cat_cols = ['store_location', 'loyalty_tier', 'order_channel', 'payment_method']
    for col in cat_cols:
        le = LabelEncoder()
        visit_df[col] = le.fit_transform(visit_df[col])
        encoders[col] = le
        
    # Target variable
    visit_df['target'] = visit_df['returned_within_14_days'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    features = ['store_location', 'customer_age', 'loyalty_tier', 'visit_number', 
                'days_since_previous_visit', 'order_channel', 'payment_method', 
                'revenue', 'discount_pct', 'wait_time_min', 'service_rating']
                
    X = visit_df[features]
    y = visit_df['target']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X, y)
    
    return rf, encoders, features

try:
    model, encoders, feature_names = load_and_train_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading dataset or training model: {e}")

# -----------------------------------------------------------------------------
# SIDEBAR - INPUT CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ☕ 365 Coffee Café")
st.sidebar.markdown("**Visit Intelligence Dashboard**")
st.sidebar.markdown("---")

st.sidebar.subheader("📍 Outlet & Customer Info")
store_location = st.sidebar.selectbox("Branch Location", ['P.E.C.H.S', 'Tipu Sultan', 'Khayaban e Seher'])
customer_age = st.sidebar.slider("Customer Age", 18, 70, 30)
loyalty_tier = st.sidebar.selectbox("Loyalty Tier", ['Basic', 'Silver', 'Gold', 'Platinum'])

st.sidebar.subheader("🧾 Visit Characteristics")
order_channel = st.sidebar.selectbox("Order Channel", ['Cafe App', 'Walk-in', 'Takeaway', 'Foodpanda', 'Careem Quik'])
payment_method = st.sidebar.selectbox("Payment Method", ['Raast', 'Credit/Debit Card', 'Cash', 'Easypaisa', 'JazzCash'])
revenue = st.sidebar.number_input("Basket Revenue (PKR)", min_value=200.0, max_value=10000.0, value=1250.0, step=50.0)
discount_pct = st.sidebar.slider("Discount Applied (%)", 0, 50, 0)

st.sidebar.subheader("⏱️ Service & History")
visit_number = st.sidebar.number_input("Customer Visit Number", min_value=1, max_value=50, value=3)
days_since_prev = st.sidebar.number_input("Days Since Previous Visit", min_value=-1, max_value=180, value=5)
wait_time = st.sidebar.slider("Wait Time (Minutes)", 1.0, 15.0, 4.2)
service_rating = st.sidebar.slider("Service Rating (1-5)", 1, 5, 4)

# -----------------------------------------------------------------------------
# MAIN CONTENT AREA
# -----------------------------------------------------------------------------
st.title("Customer Return Prediction System")
st.markdown("Predicting whether a customer visit will result in a return within **14 days** based on real operational telemetry across Karachi outlets.")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Visit Profile Summary")
    summary_data = {
        "Parameter": ["Branch", "Customer Age", "Loyalty Tier", "Order Channel", "Payment", "Basket Revenue", "Wait Time", "Service Rating"],
        "Value": [store_location, f"{customer_age} yrs", loyalty_tier, order_channel, payment_method, f"PKR {revenue:,.1f}", f"{wait_time} min", f"{service_rating} / 5 ⭐"]
    }
    st.table(pd.DataFrame(summary_data))

with col2:
    st.subheader("🎯 Real-Time Model Inference")
    
    if st.button("Predict Return Likelihood"):
        if model_loaded:
            # Prepare input array
            input_data = pd.DataFrame([[
                encoders['store_location'].transform([store_location])[0],
                customer_age,
                encoders['loyalty_tier'].transform([loyalty_tier])[0],
                visit_number,
                days_since_prev,
                encoders['order_channel'].transform([order_channel])[0],
                encoders['payment_method'].transform([payment_method])[0],
                revenue,
                discount_pct,
                wait_time,
                service_rating
            ]], columns=feature_names)
            
            prob = model.predict_proba(input_data)[0][1]
            prediction = model.predict(input_data)[0]
            
            # Display Results Cards
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                status_color = "#10B981" if prediction == 1 else "#EF4444"
                status_text = "LIKELY TO RETURN" if prediction == 1 else "UNLIKELY TO RETURN"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Status</div>
                    <div class="metric-value" style="color: {status_color}; font-size: 24px; margin-top: 10px;">{status_text}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Return Probability</div>
                    <div class="metric-value">{prob * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Actionable Recommendation based on prediction & wait time
            st.markdown("### 💡 Operational Recommendation")
            if prediction == 1:
                st.success(f"**High Retention Potential**: This visit exhibits strong repeat patterns. Maintain high service quality at **{store_location}** and encourage loyalty app engagement.")
            else:
                if wait_time > 6.0:
                    st.warning(f"**Action Required**: Elevated wait time ({wait_time} mins) is harming retention. Streamline bar operations at **{store_location}**.")
                else:
                    st.info(f"**Opportunity**: Consider offering a targeted loyalty promotion or digital discount voucher to incentivize a return within 14 days.")
        else:
            st.error("Model not available.")

# -----------------------------------------------------------------------------
# FOOTER DISCLAIMER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-box">
    <b>Note:</b> This prediction score is generated using a Random Forest classification model trained on 365 Coffee Café operational logs. It serves as a decision-support estimate rather than an absolute guarantee.
</div>
""", unsafe_allow_html=True)