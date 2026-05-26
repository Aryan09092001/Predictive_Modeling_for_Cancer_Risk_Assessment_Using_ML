# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG — must be the first Streamlit command
# ============================================================
st.set_page_config(
    page_title="Cancer Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — modern look
# ============================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffd93d 0%, #f6b93b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
    }
    .risk-low {
        background: linear-gradient(135deg, #6bcf7f 0%, #4caf50 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
    }
    [data-testid="stSidebar"] {
        min-width: 350px;
        max-width: 400px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('model_xgb_new.pkl')
    le = joblib.load('label_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, le, feature_names

model, le, FEATURE_NAMES = load_artifacts()

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">🩺 Cancer Risk Level Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered risk assessment using class-weighted XGBoost</div>', unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def preprocess_input(df):
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Missing {len(missing)} columns — filling with zeros: {missing}")
        for c in missing:
            df[c] = 0
    df = df[FEATURE_NAMES].copy()
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

def get_risk_color(label):
    return {'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#6bcf7f'}.get(label, '#6c757d')

def get_risk_class(label):
    return {'High': 'risk-high', 'Medium': 'risk-medium', 'Low': 'risk-low'}.get(label, 'risk-low')

def plot_probability_bars(probs, classes):
    colors = [get_risk_color(c) for c in classes]
    fig = go.Figure(go.Bar(
        x=probs * 100,
        y=classes,
        orientation='h',
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='auto',
        textfont=dict(size=14, color='white')
    ))
    fig.update_layout(
        title="Risk Level Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="",
        height=300,
        showlegend=False,
        xaxis=dict(range=[0, 100]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def plot_gauge(high_prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=high_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "High Risk Probability (%)", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#764ba2"},
            'steps': [
                {'range': [0, 33], 'color': '#d4edda'},
                {'range': [33, 66], 'color': '#fff3cd'},
                {'range': [66, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ============================================================
# SIDEBAR — All inputs go here
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/medical-doctor.png", width=80)
    st.title("⚙️ Patient Input")

    option = st.radio(
        "Prediction Mode",
        ("👤 Manual Input", "📂 Upload CSV"),
        index=0
    )

    st.markdown("---")

    if "Manual" in option:
        st.markdown("### 📝 Patient Features")
        st.caption("Fill in the details below")

        input_data = {}
        for feat in FEATURE_NAMES:
            if feat.lower() == 'age':
                val = st.number_input(f"🎂 {feat}", min_value=0, max_value=120, value=50, step=1)
            elif feat.lower() == 'bmi':
                val = st.number_input(f"⚖️ {feat}", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
            elif feat.lower() == 'gender':
                val = st.selectbox(f"⚧ {feat} (0=Female, 1=Male)", [0, 1])
            else:
                val = st.number_input(feat, value=0.0, step=1.0)
            input_data[feat] = float(val)

        st.markdown("---")
        predict_btn = st.button("🔍 Predict Risk Level", use_container_width=True, type="primary")
    else:
        st.markdown("### 📂 Upload Patient CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        with st.expander("📋 Required columns"):
            st.code(", ".join(FEATURE_NAMES))

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info(
        "Uses an **Optuna-tuned, class-weighted XGBoost** model "
        f"with {len(FEATURE_NAMES)} patient features."
    )

# ============================================================
# MAIN AREA — Results display
# ============================================================

if "Manual" in option:
    if predict_btn:
        X_single = pd.DataFrame([input_data])
        X_proc = preprocess_input(X_single)
        pred_enc = model.predict(X_proc)[0]
        probs = model.predict_proba(X_proc)[0]
        pred = le.inverse_transform([pred_enc])[0]

        st.markdown("## 📊 Prediction Results")

        risk_class = get_risk_class(pred)
        st.markdown(
            f'<div class="{risk_class}">Predicted Risk Level: <strong>{pred.upper()}</strong></div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            sorted_idx = np.argsort(probs)[::-1]
            sorted_classes = le.classes_[sorted_idx]
            sorted_probs = probs[sorted_idx]
            st.plotly_chart(plot_probability_bars(sorted_probs, sorted_classes), use_container_width=True)

        with col2:
            high_idx = list(le.classes_).index('High') if 'High' in le.classes_ else 0
            high_prob = probs[high_idx]
            st.plotly_chart(plot_gauge(high_prob), use_container_width=True)

        st.markdown("### 📈 Detailed Probabilities")
        metric_cols = st.columns(len(le.classes_))
        for i, cls in enumerate(le.classes_):
            with metric_cols[i]:
                st.metric(label=f"{cls} Risk", value=f"{probs[i]*100:.1f}%")

        st.markdown("### 💡 Clinical Interpretation")
        if pred == 'High':
            st.error("🚨 **High risk detected.** Immediate clinical follow-up is strongly recommended. Consider further diagnostic testing and specialist consultation.")
        elif pred == 'Medium':
            st.warning("⚠️ **Medium risk detected.** Routine monitoring and lifestyle adjustments recommended. Schedule a follow-up appointment.")
        else:
            st.success("✅ **Low risk detected.** Continue regular health check-ups and maintain a healthy lifestyle.")
    else:
        st.info("👈 Fill in the patient features in the sidebar and click **Predict Risk Level** to see results.")

        st.markdown("### 📋 Features Used by the Model")
        feat_cols = st.columns(3)
        for i, feat in enumerate(FEATURE_NAMES):
            with feat_cols[i % 3]:
                st.markdown(f"- {feat}")

else:
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(input_df)} rows from CSV")

        with st.expander("👀 Preview uploaded data"):
            st.dataframe(input_df.head(10), use_container_width=True)

        X = preprocess_input(input_df)
        preds_enc = model.predict(X)
        probs = model.predict_proba(X)
        preds = le.inverse_transform(preds_enc)

        result = input_df.copy()
        result['Predicted_Risk_Level'] = preds
        for i, cls in enumerate(le.classes_):
            result[f'prob_{cls}'] = probs[:, i]

        st.markdown("### 📊 Batch Summary")
        risk_counts = pd.Series(preds).value_counts()
        summary_cols = st.columns(len(le.classes_) + 1)

        with summary_cols[0]:
            st.metric("Total Patients", len(preds))

        for i, cls in enumerate(le.classes_):
            with summary_cols[i + 1]:
                count = risk_counts.get(cls, 0)
                pct = count / len(preds) * 100
                st.metric(f"{cls} Risk", f"{count}", f"{pct:.1f}%")

        st.markdown("### 📈 Risk Distribution")
        col1, col2 = st.columns(2)

        with col1:
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#6bcf7f'},
                title="Risk Level Distribution"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_bar = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                color=risk_counts.index,
                color_discrete_map={'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#6bcf7f'},
                title="Patient Count by Risk Level",
                labels={'x': 'Risk Level', 'y': 'Number of Patients'}
            )
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### 📋 Full Results")
        st.dataframe(result, use_container_width=True, height=400)

        st.download_button(
            label="⬇️ Download Predictions (CSV)",
            data=result.to_csv(index=False),
            file_name="cancer_risk_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("👈 Upload a CSV file in the sidebar to get batch predictions.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "🧬 **Model:** Optuna-tuned class-weighted XGBoost  |  "
    f"**Features:** {len(FEATURE_NAMES)}  |  "
    "**Note:** This tool is for educational purposes only and is not a substitute for professional medical advice."
)