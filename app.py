"""
Android Adware Detection System with Explainable AI
Version 2.2 - Updated for Computers & Security Submission
Paper: Explainable Machine Learning for Android Adware Detection
Authors: Parisa Hajibabaee, Karim Elish, Masoud Malekzadeh, Hamoud Aljamaan
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Android Adware Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .adware-card {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .benign-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS
# ==========================================

# Final 22 features used in paper
FEATURE_NAMES = [
    'network_op', 'sqllite_op', 'fileio_op', 'no_action', 'noc', 'dit',
    'lcom', 'cbo', 'ppiv', 'apd', 'start_activities', 'start_activity',
    'start_service', 'start_action_mode', 'start_activity_result',
    'start_activity_from_frag', 'start_activity_needed',
    'start_intent_for_result', 'start_search', 'methods_per_class',
    'bytecode_per_method', 'cyclomatic'
]

FEATURE_DESCRIPTIONS = {
    'cyclomatic': 'McCabe cyclomatic complexity — higher values indicate more complex, typically legitimate apps (benign avg: 38,606 vs adware avg: 5,754)',
    'network_op': 'Network operation calls — high values indicate frequent ad server communication (adware avg: 3.89 vs benign avg: 0.26)',
    'fileio_op': 'File I/O operation calls — file system access patterns (benign avg: 163 vs adware avg: 54)',
    'sqllite_op': 'SQLite database operations — local data storage behavior',
    'no_action': 'Empty exception handlers — code sophistication indicator (benign avg: 831 vs adware avg: 183)',
    'cbo': 'Coupling Between Objects — class interdependence metric (Chidamber & Kemerer suite)',
    'lcom': 'Lack of Cohesion in Methods — OOP design quality metric',
    'dit': 'Depth of Inheritance Tree — class hierarchy complexity',
    'noc': 'Number of Children — inheritance breadth measure',
    'ppiv': 'Proportion of Public Instance Variables — encapsulation indicator',
    'apd': 'Average Parameter Density — method complexity indicator',
    'start_service': 'startService() invocations — background service activity',
    'start_activity': 'startActivity() invocations — UI navigation calls',
    'start_activities': 'startActivities() invocations — multiple activity launches',
    'start_action_mode': 'startActionMode() invocations — contextual action calls',
    'start_activity_result': 'startActivityForResult() invocations — result-based navigation',
    'start_activity_from_frag': 'startActivityFromFragment() invocations',
    'start_activity_needed': 'startActivityIfNeeded() invocations',
    'start_intent_for_result': 'startIntentSenderForResult() invocations',
    'start_search': 'startSearch() invocations — search functionality calls',
    'methods_per_class': 'Average number of methods per class',
    'bytecode_per_method': 'Average bytecode instructions per method'
}

# Typical values from paper (Table 3)
TYPICAL_VALUES = {
    'benign': {
        'cyclomatic': 38606, 'network_op': 0.26, 'fileio_op': 163,
        'no_action': 831, 'cbo': 0.63, 'lcom': 56.54,
        'start_service': 8.93, 'sqllite_op': 111.06,
        'bytecode_per_method': 18.93, 'dit': 1.59
    },
    'adware': {
        'cyclomatic': 5754, 'network_op': 3.89, 'fileio_op': 54,
        'no_action': 183, 'cbo': 0.67, 'lcom': 67.88,
        'start_service': 4.74, 'sqllite_op': 29.51,
        'bytecode_per_method': 25.56, 'dit': 1.22
    }
}

# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource
def load_model():
    """Load the trained LightGBM model with scaler and feature names."""
    # Try corrected model first, then fallback
    model_files = [
        'adware_model_22features_CORRECTED.joblib',
        'adware_model_22features.joblib'
    ]
    
    for filename in model_files:
        if os.path.exists(filename):
            try:
                package = joblib.load(filename)
                if isinstance(package, dict):
                    model = package.get('model')
                    scaler = package.get('scaler')
                    features = package.get('feature_names', FEATURE_NAMES)
                    return model, scaler, features
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
                continue
    
    return None, None, FEATURE_NAMES

MODEL, SCALER, LOADED_FEATURES = load_model()

if LOADED_FEATURES is not None:
    FEATURE_NAMES = LOADED_FEATURES

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_prediction(features_df):
    """Run adware detection on input features."""
    if MODEL is None:
        return None, None
    
    available = [f for f in FEATURE_NAMES if f in features_df.columns]
    if not available:
        return None, None
    
    X = features_df[available].copy()
    
    if SCALER is not None:
        X_scaled = SCALER.transform(X)
    else:
        X_scaled = X.values
    
    pred = MODEL.predict(X_scaled)
    prob = MODEL.predict_proba(X_scaled) if hasattr(MODEL, 'predict_proba') \
           else np.array([[1-p, p] for p in pred])
    
    return pred, prob


def create_gauge(value, title, color):
    """Create a gauge chart for confidence/risk display."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        number={'suffix': '%', 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': '#e8f5e9'},
                {'range': [33, 66], 'color': '#fff3e0'},
                {'range': [66, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_radar_chart(features_df, idx=0):
    """Create radar chart comparing app profile to typical benign/adware."""
    key_features = ['cyclomatic', 'network_op', 'fileio_op',
                    'no_action', 'cbo', 'start_service']
    available = [f for f in key_features if f in features_df.columns]
    
    if not available:
        return None
    
    sample_vals = features_df[available].iloc[idx].values
    
    # Normalize values (0-1 scale based on benign typical max)
    max_vals = {
        'cyclomatic': 50000, 'network_op': 20, 'fileio_op': 300,
        'no_action': 1000, 'cbo': 1, 'start_service': 20
    }
    
    def normalize(vals, feats):
        return [min(1, abs(vals[i]) / max_vals.get(feats[i], 1))
                for i in range(len(feats))]
    
    sample_norm = normalize(sample_vals, available)
    
    benign_norm = normalize(
        [TYPICAL_VALUES['benign'].get(f, 0) for f in available], available
    )
    adware_norm = normalize(
        [TYPICAL_VALUES['adware'].get(f, 0) for f in available], available
    )
    
    # Close the radar
    theta = available + [available[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=sample_norm + [sample_norm[0]],
        theta=theta,
        fill='toself',
        name='This App',
        line_color='#1E88E5',
        fillcolor='rgba(30, 136, 229, 0.3)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=benign_norm + [benign_norm[0]],
        theta=theta,
        fill='toself',
        name='Typical Benign',
        line_color='#44aa44',
        fillcolor='rgba(68, 170, 68, 0.15)',
        line=dict(dash='dash')
    ))
    fig.add_trace(go.Scatterpolar(
        r=adware_norm + [adware_norm[0]],
        theta=theta,
        fill='toself',
        name='Typical Adware',
        line_color='#ff4444',
        fillcolor='rgba(255, 68, 68, 0.15)',
        line=dict(dash='dot')
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Feature Profile vs Typical Patterns (Normalized)',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2)
    )
    return fig


def get_counterfactual_suggestions(features_df, pred, idx=0):
    """Generate counterfactual explanations."""
    suggestions = []
    sample = features_df.iloc[idx]
    
    if pred[idx] == 1:  # Classified as adware
        if 'cyclomatic' in sample.index and sample['cyclomatic'] < 15000:
            suggestions.append({
                'Feature': 'cyclomatic',
                'Current Value': f"{sample['cyclomatic']:.0f}",
                'Suggested Change': '> 15,000',
                'Reason': 'Benign apps avg 38,606 — low complexity is a key adware indicator'
            })
        if 'network_op' in sample.index and sample['network_op'] > 2:
            suggestions.append({
                'Feature': 'network_op',
                'Current Value': f"{sample['network_op']:.1f}",
                'Suggested Change': '< 2',
                'Reason': 'Benign apps avg 0.26 — high network ops indicate ad communication'
            })
        if 'fileio_op' in sample.index and sample['fileio_op'] < 100:
            suggestions.append({
                'Feature': 'fileio_op',
                'Current Value': f"{sample['fileio_op']:.0f}",
                'Suggested Change': '> 100',
                'Reason': 'Benign apps avg 163 — low file I/O typical of adware'
            })
        if 'no_action' in sample.index and sample['no_action'] < 400:
            suggestions.append({
                'Feature': 'no_action',
                'Current Value': f"{sample['no_action']:.0f}",
                'Suggested Change': '> 400',
                'Reason': 'Benign apps avg 831 — more exception handlers indicate richer code'
            })
        if not suggestions:
            suggestions.append({
                'Feature': 'Multiple features',
                'Current Value': 'N/A',
                'Suggested Change': 'See SHAP values',
                'Reason': 'Review SHAP contributions for targeted suggestions'
            })
    else:  # Benign
        suggestions.append({
            'Feature': 'N/A',
            'Current Value': 'N/A',
            'Suggested Change': 'N/A',
            'Reason': '✅ App classified as BENIGN — no changes needed'
        })
    
    return suggestions


def get_llm_response(prompt, provider, api_key, context=""):
    """Get response from selected LLM provider."""
    system = (
        "You are an Android security analyst specializing in adware detection. "
        "You analyze results from a LightGBM-based detection system with SHAP explainability. "
        "Provide clear, actionable, and technically accurate insights."
    )
    full_prompt = f"{context}\n\n{prompt}" if context else prompt
    
    try:
        if provider == "Anthropic Claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                system=system,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.content[0].text
        
        elif provider == "OpenAI GPT-4":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1500
            )
            return response.choices[0].message.content
        
        elif provider == "Google Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            return model.generate_content(
                f"{system}\n\n{full_prompt}"
            ).text
    
    except Exception as e:
        return f"⚠️ Error: {str(e)}\n\nPlease check your API key and try again."


# ==========================================
# MAIN APP
# ==========================================
def main():
    
    # Header
    st.markdown(
        "<h1 style='text-align:center'>🛡️ Android Adware Detection System</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;color:gray;font-size:16px'>"
        "Powered by LightGBM + SHAP Explainable AI | 99.32% Accuracy</p>",
        unsafe_allow_html=True
    )
    
    # ==========================================
    # SIDEBAR
    # ==========================================
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/android-os.png", width=80)
        st.markdown("## ⚙️ Configuration")
        
        llm_provider = st.selectbox(
            "Select AI Provider",
            ["Anthropic Claude", "OpenAI GPT-4", "Google Gemini"]
        )
        api_key = st.text_input(
            f"{llm_provider} API Key",
            type="password",
            placeholder="Enter your API key..."
        )
        
        if api_key:
            st.success(f"✓ {llm_provider} configured")
        else:
            st.info("Add API key to enable AI chat features")
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        if MODEL is not None:
            st.success("✓ Model Loaded Successfully")
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", "99.32%")
            col2.metric("F1-Score", "99.42%")
            col1.metric("ROC-AUC", "99.91%")
            col2.metric("Features", f"{len(FEATURE_NAMES)}")
            st.info("Algorithm: LightGBM\n(Gradient Boosting)")
        else:
            st.error("✗ Model not loaded")
            st.info("Ensure adware_model_22features.joblib is in the app directory")
        
        st.markdown("---")
        st.markdown("### 🔗 Resources")
        st.markdown("[📂 GitHub Repository](https://github.com/parisahjb/Cybersecurity-Adware-Detection)")
        st.markdown("[📥 Feature Extraction Tool](https://github.com/sealuzh/user_quality)")
        
        st.markdown("---")
        st.markdown("### 📄 Citation")
        st.markdown("""
        <small>Hajibabaee, P., Elish, K., Malekzadeh, M., & Aljamaan, H. (2025). 
        <em>Explainable Machine Learning for Android Adware Detection.</em> 
        # Computers & Security.</small>
        """, unsafe_allow_html=True)
    
    # ==========================================
    # SESSION STATE
    # ==========================================
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'chat' not in st.session_state:
        st.session_state.chat = []
    
    # ==========================================
    # TABS
    # ==========================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Detection",
        "📊 Explainability",
        "🤖 AI Chat",
        "📋 Batch Analysis",
        "📄 Report"
    ])
    
    # ==========================================
    # TAB 1: DETECTION
    # ==========================================
    with tab1:
        st.markdown("### 📱 Input Data")
        st.markdown(
            '<div class="info-box"><strong>ℹ️ Note:</strong> Features are extracted '
            'from Android APK files using the '
            '<a href="https://github.com/sealuzh/user_quality" target="_blank">'
            'Android Quality Metrics Tool</a> by Grano et al.</div>',
            unsafe_allow_html=True
        )
        
        input_method = st.radio(
            "Select Input Method:",
            ["📂 Upload CSV", "📊 Load Sample Data", "✏️ Manual Input"],
            horizontal=True
        )
        
        # CSV Upload
        if input_method == "📂 Upload CSV":
            uploaded = st.file_uploader(
                "Upload CSV file with extracted features",
                type=['csv'],
                help="CSV should contain the 22 static analysis features"
            )
            if uploaded:
                st.session_state.features = pd.read_csv(uploaded)
                st.success(f"✓ Loaded {len(st.session_state.features)} sample(s)")
        
        # Sample Data
        elif input_method == "📊 Load Sample Data":
            st.markdown("Load pre-extracted sample data for demonstration.")
            if st.button("📂 Load Sample Data", type="secondary"):
                if os.path.exists('sample_test_data.csv'):
                    st.session_state.features = pd.read_csv('sample_test_data.csv')
                    st.success(f"✓ Loaded {len(st.session_state.features)} sample(s)")
                else:
                    st.error("sample_test_data.csv not found in repository")
        
        # Manual Input
        elif input_method == "✏️ Manual Input":
            st.markdown("#### Enter Feature Values")
            st.markdown(
                "*Enter values for each feature. "
                "Refer to descriptions for guidance.*"
            )
            
            manual_data = {}
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📐 Code Complexity Metrics**")
                manual_data['cyclomatic'] = st.number_input(
                    'cyclomatic', value=5000.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['cyclomatic']
                )
                manual_data['methods_per_class'] = st.number_input(
                    'methods_per_class', value=5.0, format="%.2f",
                    help=FEATURE_DESCRIPTIONS['methods_per_class']
                )
                manual_data['bytecode_per_method'] = st.number_input(
                    'bytecode_per_method', value=20.0, format="%.2f",
                    help=FEATURE_DESCRIPTIONS['bytecode_per_method']
                )
                st.markdown("**🏗️ OO Design Metrics**")
                manual_data['lcom'] = st.number_input(
                    'lcom', value=30.0, format="%.2f",
                    help=FEATURE_DESCRIPTIONS['lcom']
                )
                manual_data['cbo'] = st.number_input(
                    'cbo', value=0.5, format="%.3f",
                    help=FEATURE_DESCRIPTIONS['cbo']
                )
                manual_data['dit'] = st.number_input(
                    'dit', value=2.0, format="%.2f",
                    help=FEATURE_DESCRIPTIONS['dit']
                )
                manual_data['noc'] = st.number_input(
                    'noc', value=1.0, format="%.2f",
                    help=FEATURE_DESCRIPTIONS['noc']
                )
                manual_data['ppiv'] = st.number_input(
                    'ppiv', value=0.0, format="%.2f",
                    help=FEATURE_DESCRIPTIONS['ppiv']
                )
                manual_data['apd'] = st.number_input(
                    'apd', value=0.0, format="%.2f",
                    help=FEATURE_DESCRIPTIONS['apd']
                )
            
            with col2:
                st.markdown("**⚙️ Behavioral Features**")
                manual_data['network_op'] = st.number_input(
                    'network_op', value=4.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['network_op']
                )
                manual_data['fileio_op'] = st.number_input(
                    'fileio_op', value=50.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['fileio_op']
                )
                manual_data['sqllite_op'] = st.number_input(
                    'sqllite_op', value=0.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['sqllite_op']
                )
                manual_data['no_action'] = st.number_input(
                    'no_action', value=200.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['no_action']
                )
            
            with col3:
                st.markdown("**📡 Android API Calls**")
                manual_data['start_activity'] = st.number_input(
                    'start_activity', value=5.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['start_activity']
                )
                manual_data['start_activities'] = st.number_input(
                    'start_activities', value=0.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['start_activities']
                )
                manual_data['start_service'] = st.number_input(
                    'start_service', value=3.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['start_service']
                )
                manual_data['start_action_mode'] = st.number_input(
                    'start_action_mode', value=0.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['start_action_mode']
                )
                manual_data['start_activity_result'] = st.number_input(
                    'start_activity_result', value=0.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['start_activity_result']
                )
                manual_data['start_activity_from_frag'] = st.number_input(
                    'start_activity_from_frag', value=0.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['start_activity_from_frag']
                )
                manual_data['start_activity_needed'] = st.number_input(
                    'start_activity_needed', value=0.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['start_activity_needed']
                )
                manual_data['start_intent_for_result'] = st.number_input(
                    'start_intent_for_result', value=0.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['start_intent_for_result']
                )
                manual_data['start_search'] = st.number_input(
                    'start_search', value=0.0, format="%.1f",
                    help=FEATURE_DESCRIPTIONS['start_search']
                )
            
            if st.button("✅ Confirm Manual Input", type="primary"):
                st.session_state.features = pd.DataFrame([manual_data])
                st.success("✓ Manual input ready for analysis")
        
        # Detection Button
        st.markdown("---")
        if st.session_state.features is not None:
            st.info(
                f"📊 **{len(st.session_state.features)} sample(s) ready for analysis**"
            )
            
            with st.expander("👁️ Preview Data"):
                st.dataframe(
                    st.session_state.features.head(5),
                    use_container_width=True
                )
            
            if st.button(
                "🔍 RUN ADWARE DETECTION",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Analyzing application features..."):
                    pred, prob = get_prediction(st.session_state.features)
                
                if pred is not None:
                    st.session_state.results = {'pred': pred, 'prob': prob}
                    st.success(
                        f"✓ Detection complete! "
                        f"{'🚨 ADWARE DETECTED' if pred[0] == 1 else '✅ BENIGN'}"
                    )
                else:
                    st.error("Detection failed. Check model and feature names.")
        
        # Results Display
        if st.session_state.results is not None:
            st.markdown("---")
            st.markdown("### 🎯 Detection Results")
            
            pred = st.session_state.results['pred']
            prob = st.session_state.results['prob']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if pred[0] == 1:
                    st.markdown(
                        '<div class="adware-card">🚨 ADWARE DETECTED</div>',
                        unsafe_allow_html=True
                    )
                    risk_level = (
                        "HIGH" if prob[0][1] >= 0.8 else
                        "MEDIUM" if prob[0][1] >= 0.5 else "LOW"
                    )
                    st.markdown(f"<center><b>Risk Level: {risk_level}</b></center>",
                               unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="benign-card">✅ BENIGN APPLICATION</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        "<center><b>No adware behavior detected</b></center>",
                        unsafe_allow_html=True
                    )
            
            with col2:
                st.plotly_chart(
                    create_gauge(max(prob[0]) * 100, "Confidence", "#1E88E5"),
                    use_container_width=True
                )
            
            with col3:
                color = "#ff4444" if prob[0][1] > 0.5 else "#44aa44"
                st.plotly_chart(
                    create_gauge(prob[0][1] * 100, "Adware Probability", color),
                    use_container_width=True
                )
            
            # Batch summary
            if len(pred) > 1:
                st.markdown("### 📊 Batch Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Samples", len(pred))
                col2.metric("Adware Detected", int(sum(pred)),
                           delta=f"{sum(pred)/len(pred)*100:.1f}%")
                col3.metric("Benign", int(len(pred) - sum(pred)))
                col4.metric("Detection Rate",
                           f"{sum(pred)/len(pred)*100:.1f}%")
            
            # Feature importance
            if hasattr(MODEL, 'feature_importances_'):
                st.markdown("### 📈 Feature Importance")
                available = [f for f in FEATURE_NAMES
                            if f in st.session_state.features.columns]
                imp_vals = MODEL.feature_importances_[:len(available)]
                imp_df = pd.DataFrame({
                    'Feature': available[:len(imp_vals)],
                    'Importance': imp_vals
                }).sort_values('Importance', ascending=True).tail(15)
                
                fig = px.bar(
                    imp_df, x='Importance', y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='RdYlGn_r',
                    title='LightGBM Feature Importance (Top 15)'
                )
                fig.update_layout(height=450, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================
    # TAB 2: EXPLAINABILITY
    # ==========================================
    with tab2:
        st.markdown("### 📊 SHAP-Based Explainability")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) values quantify each "
            "feature's contribution to the classification decision, "
            "providing both local and global interpretability."
        )
        
        if st.session_state.results is None:
            st.info("👆 Run detection first (Detection tab) to see explanations")
        else:
            pred = st.session_state.results['pred']
            prob = st.session_state.results['prob']
            features = st.session_state.features
            
            # Sample selector
            if len(features) > 1:
                sample_idx = st.selectbox(
                    "Select Sample to Analyze:",
                    range(len(features)),
                    format_func=lambda x: (
                        f"Sample {x+1} — "
                        f"{'🚨 ADWARE' if pred[x] == 1 else '✅ BENIGN'} "
                        f"({prob[x][1]*100:.1f}% adware probability)"
                    )
                )
            else:
                sample_idx = 0
            
            col1, col2 = st.columns(2)
            
            # SHAP Waterfall
            with col1:
                st.markdown("#### SHAP Feature Contributions")
                st.markdown(
                    "🔴 **Red bars** → Push toward **ADWARE** | "
                    "🟢 **Green bars** → Push toward **BENIGN**"
                )
                try:
                    import shap
                    available = [f for f in FEATURE_NAMES
                                if f in features.columns]
                    X = features[available]
                    
                    if SCALER:
                        X_scaled = pd.DataFrame(
                            SCALER.transform(X), columns=available
                        )
                    else:
                        X_scaled = X.reset_index(drop=True)
                    
                    explainer = shap.TreeExplainer(MODEL)
                    shap_vals = explainer.shap_values(X_scaled)
                    
                    if isinstance(shap_vals, list):
                        sv = shap_vals[1][sample_idx]
                    else:
                        sv = shap_vals[sample_idx]
                    
                    # Top 10 by absolute value
                    top_idx = np.argsort(np.abs(sv))[::-1][:10]
                    top_features = [available[i] for i in top_idx]
                    top_values = [sv[i] for i in top_idx]
                    
                    fig = go.Figure(go.Bar(
                        y=top_features[::-1],
                        x=top_values[::-1],
                        orientation='h',
                        marker_color=[
                            '#ff4444' if s > 0 else '#44aa44'
                            for s in top_values[::-1]
                        ],
                        text=[f"{v:.3f}" for v in top_values[::-1]],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title='SHAP Values — Top 10 Features',
                        xaxis_title='SHAP Value (Impact on Adware Prediction)',
                        height=420,
                        margin=dict(l=10, r=60, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show mean absolute SHAP
                    mean_abs = np.abs(sv).sum()
                    top_contribution = (
                        np.abs(sv[top_idx[0]]) / mean_abs * 100
                        if mean_abs > 0 else 0
                    )
                    st.caption(
                        f"Top feature ({top_features[0]}) accounts for "
                        f"{top_contribution:.1f}% of total SHAP magnitude"
                    )
                    
                except ImportError:
                    st.warning(
                        "SHAP library not installed. "
                        "Run: `pip install shap`"
                    )
                except Exception as e:
                    st.warning(f"SHAP calculation error: {e}")
            
            # Radar Chart
            with col2:
                st.markdown("#### Feature Profile Comparison")
                st.markdown(
                    "Compares this app's feature profile against "
                    "typical benign and adware patterns from our dataset."
                )
                radar_fig = create_radar_chart(features, sample_idx)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
                else:
                    st.warning("Radar chart unavailable — missing key features")
            
            # Counterfactual Analysis
            st.markdown("---")
            st.markdown("#### 🔄 Counterfactual Analysis")
            st.markdown(
                "*What minimal changes to feature values would alter "
                "the classification decision?*"
            )
            suggestions = get_counterfactual_suggestions(
                features, pred, sample_idx
            )
            if suggestions:
                st.dataframe(
                    pd.DataFrame(suggestions),
                    use_container_width=True,
                    hide_index=True
                )
            
            # Feature Details Table
            st.markdown("---")
            st.markdown("#### 📋 Feature Details")
            sample = features.iloc[sample_idx]
            table_data = []
            for f in FEATURE_NAMES:
                if f in sample.index:
                    table_data.append({
                        'Feature': f,
                        'Value': f"{float(sample[f]):.4f}",
                        'Description': FEATURE_DESCRIPTIONS.get(f, '—'),
                        'Benign Avg': TYPICAL_VALUES['benign'].get(f, '—'),
                        'Adware Avg': TYPICAL_VALUES['adware'].get(f, '—')
                    })
            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True
            )
    
    # ==========================================
    # TAB 3: AI CHAT
    # ==========================================
    with tab3:
        st.markdown(f"### 🤖 AI Security Analyst — {llm_provider}")
        st.markdown(
            "Ask questions about the detection results, feature values, "
            "or Android adware behavior."
        )
        
        if not api_key:
            st.warning(
                "⚠️ Enter your API key in the sidebar to enable AI features. "
                "Supported providers: Anthropic Claude, OpenAI GPT-4, "
                "Google Gemini."
            )
        else:
            # Build context
            context = ""
            if st.session_state.results:
                pred = st.session_state.results['pred']
                prob = st.session_state.results['prob']
                feat = st.session_state.features.iloc[0]
                context = (
                    f"Detection Result: {'ADWARE' if pred[0] == 1 else 'BENIGN'}\n"
                    f"Confidence: {max(prob[0])*100:.1f}%\n"
                    f"Adware Probability: {prob[0][1]*100:.1f}%\n"
                    f"Key Feature Values:\n"
                    f"  - cyclomatic: {feat.get('cyclomatic', 'N/A'):.0f} "
                    f"(benign avg: 38,606 | adware avg: 5,754)\n"
                    f"  - network_op: {feat.get('network_op', 'N/A'):.2f} "
                    f"(benign avg: 0.26 | adware avg: 3.89)\n"
                    f"  - fileio_op: {feat.get('fileio_op', 'N/A'):.0f} "
                    f"(benign avg: 163 | adware avg: 54)\n"
                    f"  - no_action: {feat.get('no_action', 'N/A'):.0f} "
                    f"(benign avg: 831 | adware avg: 183)\n"
                    f"Model: LightGBM, 99.32% accuracy, 22 static features"
                )
                st.markdown(
                    '<div class="info-box">💡 AI has context about your '
                    'current detection results and key feature values.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info(
                    "ℹ️ Run detection first for context-aware responses. "
                    "You can still ask general questions about adware detection."
                )
            
            # Quick action buttons
            st.markdown("#### ⚡ Quick Actions")
            col1, col2, col3 = st.columns(3)
            
            if col1.button("📝 Explain This Detection",
                          use_container_width=True):
                if st.session_state.results:
                    with st.spinner("Generating explanation..."):
                        r = get_llm_response(
                            "Explain this adware detection result in detail. "
                            "What are the key indicators and what do they mean "
                            "for this application's behavior?",
                            llm_provider, api_key, context
                        )
                        st.session_state.chat.append(
                            ("Explain This Detection", r)
                        )
                        st.rerun()
                else:
                    st.warning("Please run detection first!")
            
            if col2.button("⚠️ Security Advisory",
                          use_container_width=True):
                if st.session_state.results:
                    with st.spinner("Generating security advisory..."):
                        r = get_llm_response(
                            "Provide a comprehensive security advisory for "
                            "this application. What risks does it pose to "
                            "user privacy? What immediate actions should "
                            "users and security teams take?",
                            llm_provider, api_key, context
                        )
                        st.session_state.chat.append(("Security Advisory", r))
                        st.rerun()
                else:
                    st.warning("Please run detection first!")
            
            if col3.button("🔍 Technical Analysis",
                          use_container_width=True):
                if st.session_state.results:
                    with st.spinner("Running technical analysis..."):
                        r = get_llm_response(
                            "Provide a detailed technical analysis of the "
                            "feature values. Explain what each key feature "
                            "value indicates about the application's "
                            "implementation and behavior patterns.",
                            llm_provider, api_key, context
                        )
                        st.session_state.chat.append(
                            ("Technical Analysis", r)
                        )
                        st.rerun()
                else:
                    st.warning("Please run detection first!")
            
            # Free-form chat
            st.markdown("---")
            st.markdown("#### 💬 Ask a Question")
            q = st.text_input(
                "Type your question:",
                placeholder=(
                    "e.g., Why is cyclomatic complexity important? "
                    "What is the risk of high network operations?"
                )
            )
            if st.button("📤 Send", type="primary") and q:
                with st.spinner("Thinking..."):
                    r = get_llm_response(q, llm_provider, api_key, context)
                    st.session_state.chat.append((q, r))
                    st.rerun()
            
            # Chat history
            if st.session_state.chat:
                st.markdown("---")
                st.markdown("#### 💬 Conversation History")
                for question, answer in reversed(
                    st.session_state.chat[-5:]
                ):
                    with st.expander(f"**Q: {question[:80]}...**"
                                    if len(question) > 80
                                    else f"**Q: {question}**",
                                    expanded=True):
                        st.markdown(f"**🤖 AI:** {answer}")
                
                if st.button("🗑️ Clear Conversation"):
                    st.session_state.chat = []
                    st.rerun()
    
    # ==========================================
    # TAB 4: BATCH ANALYSIS
    # ==========================================
    with tab4:
        st.markdown("### 📋 Batch Analysis Results")
        
        if st.session_state.results is None:
            st.info("👆 Run detection first (Detection tab) to see batch results")
        else:
            pred = st.session_state.results['pred']
            prob = st.session_state.results['prob']
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Samples", len(pred))
            col2.metric(
                "Adware Detected", int(sum(pred)),
                delta=f"{sum(pred)/len(pred)*100:.1f}% of total"
            )
            col3.metric("Benign", int(len(pred) - sum(pred)))
            col4.metric(
                "Avg Confidence",
                f"{np.mean([max(p) for p in prob])*100:.1f}%"
            )
            
            # Risk breakdown
            high_risk = sum(1 for p in prob if p[1] >= 0.8)
            med_risk = sum(1 for p in prob if 0.5 <= p[1] < 0.8)
            low_risk = sum(1 for p in prob if p[1] < 0.5)
            
            st.markdown("#### Risk Distribution")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 High Risk (≥80%)", high_risk)
            col2.metric("🟡 Medium Risk (50-80%)", med_risk)
            col3.metric("🟢 Low Risk (<50%)", low_risk)
            
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(
                    values=[int(sum(pred)), int(len(pred) - sum(pred))],
                    names=['Adware', 'Benign'],
                    color_discrete_sequence=['#ff4444', '#44aa44'],
                    title='Detection Distribution',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(
                    x=[p[1] * 100 for p in prob],
                    nbins=20,
                    title='Adware Probability Distribution',
                    labels={'x': 'Adware Probability (%)'},
                    color_discrete_sequence=['#1E88E5']
                )
                fig.add_vline(x=50, line_dash='dash',
                             line_color='red', annotation_text='Threshold')
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results table
            st.markdown("#### 📊 Detailed Results")
            df = st.session_state.features.copy()
            df['Prediction'] = [
                '🚨 ADWARE' if p == 1 else '✅ BENIGN' for p in pred
            ]
            df['Adware Prob (%)'] = [f"{p[1]*100:.1f}" for p in prob]
            df['Risk Level'] = [
                'HIGH' if p[1] >= 0.8 else
                ('MEDIUM' if p[1] >= 0.5 else 'LOW')
                for p in prob
            ]
            
            display_cols = (
                ['Prediction', 'Adware Prob (%)', 'Risk Level'] +
                [c for c in ['cyclomatic', 'network_op', 'fileio_op',
                             'no_action']
                 if c in df.columns]
            )
            st.dataframe(
                df[display_cols],
                use_container_width=True,
                hide_index=True
            )
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Results (CSV)",
                csv,
                f"adware_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
    
    # ==========================================
    # TAB 5: REPORT
    # ==========================================
    with tab5:
        st.markdown("### 📄 Detection Report")
        
        if st.session_state.results is None:
            st.info("👆 Run detection first to generate a report")
        else:
            report_type = st.selectbox(
                "Select Report Type:",
                ["Executive Summary", "Technical Report", "Security Advisory"]
            )
            
            if st.button("📄 Generate Report", type="primary",
                        use_container_width=True):
                pred = st.session_state.results['pred']
                prob = st.session_state.results['prob']
                feat = st.session_state.features.iloc[0]
                
                high_risk = sum(1 for p in prob if p[1] >= 0.8)
                med_risk = sum(1 for p in prob if 0.5 <= p[1] < 0.8)
                low_risk = sum(1 for p in prob if p[1] < 0.5)
                
                report = f"""
# 🛡️ Android Adware Detection Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Report Type:** {report_type}  
**System:** Android Adware Detection v2.2 (LightGBM, 99.32% accuracy)

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| Total Samples Analyzed | {len(pred)} |
| Adware Detected | {int(sum(pred))} ({sum(pred)/len(pred)*100:.1f}%) |
| Benign Applications | {int(len(pred)-sum(pred))} ({(len(pred)-sum(pred))/len(pred)*100:.1f}%) |
| Average Confidence | {np.mean([max(p) for p in prob])*100:.1f}% |
| High Risk (≥80%) | {high_risk} |
| Medium Risk (50-80%) | {med_risk} |
| Low Risk (<50%) | {low_risk} |

---

## 🔍 Key Findings

**Primary Indicators (most discriminative features):**
- Cyclomatic Complexity: {feat.get('cyclomatic', 'N/A'):.0f} 
  (Benign avg: 38,606 | Adware avg: 5,754)
- Network Operations: {feat.get('network_op', 'N/A'):.2f} 
  (Benign avg: 0.26 | Adware avg: 3.89)
- File I/O Operations: {feat.get('fileio_op', 'N/A'):.0f} 
  (Benign avg: 163 | Adware avg: 54)

---

## ⚠️ Risk Assessment

{'🚨 **HIGH RISK:** Adware detected with high confidence. Immediate action recommended.' 
 if prob[0][1] >= 0.8 else
 '🟡 **MEDIUM RISK:** Possible adware. Manual review recommended.' 
 if prob[0][1] >= 0.5 else
 '✅ **LOW RISK:** Application appears benign.'}

---

## 📋 Recommendations

1. **Immediate:** {'Quarantine flagged applications' if sum(pred) > 0 else 'No immediate action required'}
2. **Short-term:** Monitor network activity of flagged applications
3. **Long-term:** Implement continuous scanning pipeline
4. **User Advisory:** {'Notify affected users immediately' if sum(pred) > 0 else 'Continue standard monitoring'}

---

## 🤖 Model Information

| Parameter | Value |
|-----------|-------|
| Algorithm | LightGBM (Gradient Boosting) |
| Accuracy | 99.32% |
| F1-Score | 99.42% |
| ROC-AUC | 99.91% |
| MCC | 0.9860 |
| Features | 22 static analysis features |
| Validation | 10-fold cross-validation |

---

## 📚 Reference

Hajibabaee, P., Elish, K., Malekzadeh, M., & Aljamaan, H. (2025). 
*Explainable Machine Learning for Android Adware Detection.* 
# Computers & Security.

GitHub: https://github.com/parisahjb/Cybersecurity-Adware-Detection

---
*Report generated by Android Adware Detection System v2.2*
                """
                st.markdown(report)
                
                # Download report
                st.download_button(
                    "📥 Download Report (Markdown)",
                    report,
                    f"adware_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    "text/markdown",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
