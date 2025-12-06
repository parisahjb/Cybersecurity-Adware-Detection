"""
Android Adware Detection System with Explainable AI
Version 2.0 - Streamlit Cloud Compatible
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Android Adware Detection",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

# Feature names
FEATURE_NAMES = [
    'network_op', 'sqllite_op', 'fileio_op', 'no_action', 'noc', 'dit',
    'lcom', 'cbo', 'ppiv', 'apd', 'start_activities', 'start_activity',
    'start_service', 'start_action_mode', 'start_activity_result',
    'start_activity_from_frag', 'start_activity_needed',
    'start_intent_for_result', 'start_search', 'methods_per_class',
    'bytecode_per_method', 'cyclomatic'
]

FEATURE_DESCRIPTIONS = {
    'cyclomatic': 'Code complexity - higher = more complex, typically legitimate apps',
    'network_op': 'Network operations - high values may indicate ad activity',
    'fileio_op': 'File I/O operations - file system access patterns',
    'sqllite_op': 'SQLite operations - data storage behavior',
    'no_action': 'Empty exception handlers - code quality indicator',
    'cbo': 'Coupling Between Objects - class interdependence',
    'lcom': 'Lack of Cohesion in Methods - OOP design quality',
    'dit': 'Depth of Inheritance Tree - hierarchy complexity',
    'start_service': 'Service start calls - background processing',
}

# Load model
@st.cache_resource
def load_model():
    try:
        if os.path.exists('adware_model_22features.joblib'):
            package = joblib.load('adware_model_22features.joblib')
            return package['model'], package['scaler'], package['feature_names']
    except Exception as e:
        st.error(f"Model loading error: {e}")
    return None, None, None

MODEL, SCALER, LOADED_FEATURES = load_model()

# Use loaded features or default
if LOADED_FEATURES is not None:
    FEATURE_NAMES = LOADED_FEATURES

# Helper functions
def get_prediction(features_df):
    if MODEL is None:
        return None, None
    
    available = [f for f in FEATURE_NAMES if f in features_df.columns]
    X = features_df[available].copy()
    
    if SCALER is not None:
        X_scaled = SCALER.transform(X)
    else:
        X_scaled = X.values
    
    pred = MODEL.predict(X_scaled)
    prob = MODEL.predict_proba(X_scaled) if hasattr(MODEL, 'predict_proba') else np.array([[1-p, p] for p in pred])
    
    return pred, prob

def create_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': '#e8f5e9'},
                {'range': [33, 66], 'color': '#fff3e0'},
                {'range': [66, 100], 'color': '#ffebee'}
            ]
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# LLM function
def get_llm_response(prompt, provider, api_key, context=""):
    system = "You are an Android security analyst specializing in adware detection. Provide clear insights."
    full_prompt = f"{context}\n\n{prompt}" if context else prompt
    
    try:
        if provider == "Anthropic Claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                system=system,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.content[0].text
        elif provider == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": full_prompt}],
                max_tokens=1500
            )
            return response.choices[0].message.content
        elif provider == "Google Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            return model.generate_content(f"{system}\n\n{full_prompt}").text
    except Exception as e:
        return f"Error: {str(e)}"

# Main app
def main():
    st.markdown("<h1 style='text-align:center'>🛡️ Android Adware Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray'>Powered by Machine Learning & Explainable AI</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/android-os.png", width=80)
        st.markdown("## ⚙️ Configuration")
        
        llm_provider = st.selectbox("AI Provider", ["Anthropic Claude", "OpenAI", "Google Gemini"])
        api_key = st.text_input(f"{llm_provider} API Key", type="password")
        
        if api_key:
            st.success(f"✓ {llm_provider} configured")
        
        st.markdown("---")
        st.markdown("### Model Info")
        if MODEL is not None:
            st.success("✓ Model Loaded")
            st.info(f"Features: {len(FEATURE_NAMES)}")
            st.info("Accuracy: 99.32%")
        else:
            st.error("✗ Model not loaded")
        
        st.markdown("---")
        st.markdown("[📥 Feature Extraction Tool](https://github.com/sealuzh/user_quality)")
    
    # Session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'features' not in st.session_state:
        st.session_state.features = None
    if 'chat' not in st.session_state:
        st.session_state.chat = []
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Detection", "📊 Explainability", "🤖 AI Chat", "📋 Batch", "📄 Report"])
    
    # TAB 1: Detection
    with tab1:
        st.markdown("### Input Data")
        
        col1, col2 = st.columns(2)
        with col1:
            uploaded = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded:
                st.session_state.features = pd.read_csv(uploaded)
                st.success(f"✓ Loaded {len(st.session_state.features)} samples")
        
        with col2:
            if st.button("📂 Load Sample Data"):
                if os.path.exists('sample_test_data.csv'):
                    st.session_state.features = pd.read_csv('sample_test_data.csv')
                    st.success("✓ Sample loaded")
                else:
                    st.error("Sample file not found")
        
        if st.session_state.features is not None:
            st.info(f"📊 {len(st.session_state.features)} sample(s) ready")
            
            if st.button("🔍 RUN ADWARE DETECTION", type="primary", use_container_width=True):
                pred, prob = get_prediction(st.session_state.features)
                if pred is not None:
                    st.session_state.results = {'pred': pred, 'prob': prob}
                    st.success("✓ Detection complete!")
        
        if st.session_state.results is not None:
            st.markdown("---")
            st.markdown("### Results")
            
            pred = st.session_state.results['pred']
            prob = st.session_state.results['prob']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if pred[0] == 1:
                    st.markdown('<div class="adware-card">🚨 ADWARE DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="benign-card">✅ BENIGN</div>', unsafe_allow_html=True)
            with col2:
                st.plotly_chart(create_gauge(max(prob[0])*100, "Confidence", "#1E88E5"), use_container_width=True)
            with col3:
                st.plotly_chart(create_gauge(prob[0][1]*100, "Risk", "#ff4444" if prob[0][1] > 0.5 else "#44aa44"), use_container_width=True)
            
            if len(pred) > 1:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total", len(pred))
                col2.metric("Adware", sum(pred))
                col3.metric("Benign", len(pred) - sum(pred))
                col4.metric("Rate", f"{sum(pred)/len(pred)*100:.1f}%")
            
            # Feature importance
            if hasattr(MODEL, 'feature_importances_'):
                st.markdown("### Feature Importance")
                available = [f for f in FEATURE_NAMES if f in st.session_state.features.columns]
                imp = pd.DataFrame({'Feature': available[:len(MODEL.feature_importances_)], 
                                   'Importance': MODEL.feature_importances_}).sort_values('Importance', ascending=True).tail(15)
                fig = px.bar(imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Explainability
    with tab2:
        st.markdown("### Explainability")
        if st.session_state.results is None:
            st.info("Run detection first")
        else:
            try:
                import shap
                available = [f for f in FEATURE_NAMES if f in st.session_state.features.columns]
                X = st.session_state.features[available]
                if SCALER:
                    X_scaled = pd.DataFrame(SCALER.transform(X), columns=available)
                else:
                    X_scaled = X
                
                explainer = shap.TreeExplainer(MODEL)
                shap_vals = explainer.shap_values(X_scaled)
                
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                else:
                    sv = shap_vals[0]
                
                idx = np.argsort(np.abs(sv))[::-1][:10]
                fig = go.Figure(go.Bar(
                    y=[available[i] for i in idx][::-1],
                    x=[sv[i] for i in idx][::-1],
                    orientation='h',
                    marker_color=['#ff4444' if s > 0 else '#44aa44' for s in [sv[i] for i in idx][::-1]]
                ))
                fig.update_layout(title='SHAP Values (Top 10)', height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("🔴 Red → ADWARE | 🟢 Green → BENIGN")
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")
            
            st.markdown("### Feature Details")
            sample = st.session_state.features.iloc[0]
            table = [{'Feature': f, 'Value': f"{sample[f]:.4f}" if f in sample else "N/A", 
                     'Description': FEATURE_DESCRIPTIONS.get(f, '')} for f in FEATURE_NAMES if f in sample]
            st.dataframe(pd.DataFrame(table), hide_index=True)
    
    # TAB 3: AI Chat
    with tab3:
        st.markdown(f"### 🤖 AI Analyst ({llm_provider})")
        
        if not api_key:
            st.warning("Enter API key in sidebar")
        else:
            context = ""
            if st.session_state.results:
                pred = st.session_state.results['pred'][0]
                prob = st.session_state.results['prob'][0]
                context = f"Detection: {'ADWARE' if pred == 1 else 'BENIGN'}, Confidence: {max(prob)*100:.1f}%"
            
            col1, col2, col3 = st.columns(3)
            if col1.button("📝 Explain", use_container_width=True):
                if st.session_state.results:
                    with st.spinner("..."): 
                        r = get_llm_response("Explain this detection.", llm_provider, api_key, context)
                        st.session_state.chat.append(("Explain", r))
            if col2.button("⚠️ Advisory", use_container_width=True):
                if st.session_state.results:
                    with st.spinner("..."): 
                        r = get_llm_response("Security advisory with risks.", llm_provider, api_key, context)
                        st.session_state.chat.append(("Advisory", r))
            if col3.button("🔍 Technical", use_container_width=True):
                if st.session_state.results:
                    with st.spinner("..."): 
                        r = get_llm_response("Technical analysis.", llm_provider, api_key, context)
                        st.session_state.chat.append(("Technical", r))
            
            q = st.text_input("Ask a question...")
            if st.button("Send") and q:
                with st.spinner("..."): 
                    r = get_llm_response(q, llm_provider, api_key, context)
                    st.session_state.chat.append((q, r))
            
            for question, answer in reversed(st.session_state.chat[-5:]):
                st.markdown(f"**You:** {question}")
                st.markdown(f"**AI:** {answer}")
                st.markdown("---")
    
    # TAB 4: Batch
    with tab4:
        st.markdown("### Batch Analysis")
        if st.session_state.results is None:
            st.info("Run detection first")
        else:
            pred = st.session_state.results['pred']
            prob = st.session_state.results['prob']
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(values=[sum(pred), len(pred)-sum(pred)], names=['Adware', 'Benign'],
                            color_discrete_sequence=['#ff4444', '#44aa44'])
                st.plotly_chart(fig)
            with col2:
                fig = px.histogram(x=[p[1]*100 for p in prob], nbins=20, labels={'x': 'Probability %'})
                st.plotly_chart(fig)
            
            df = st.session_state.features.copy()
            df['Prediction'] = ['ADWARE' if p == 1 else 'BENIGN' for p in pred]
            df['Probability'] = [f"{p[1]*100:.1f}%" for p in prob]
            st.dataframe(df[['Prediction', 'Probability'] + [c for c in ['cyclomatic', 'network_op'] if c in df.columns]], hide_index=True)
            
            st.download_button("📥 Download", df.to_csv(index=False), "results.csv")
    
    # TAB 5: Report
    with tab5:
        if st.session_state.results and st.button("📄 Generate Report"):
            pred = st.session_state.results['pred']
            prob = st.session_state.results['prob']
            st.markdown(f"""
# Adware Detection Report
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

| Metric | Value |
|--------|-------|
| Total | {len(pred)} |
| Adware | {sum(pred)} ({sum(pred)/len(pred)*100:.1f}%) |
| Benign | {len(pred)-sum(pred)} |
| Confidence | {np.mean([max(p) for p in prob])*100:.1f}% |

## Recommendations
1. Quarantine high-risk apps
2. Review medium-risk apps manually
3. Monitor network activity
            """)

if __name__ == "__main__":
    main()
```

---

**Steps:**

1. Go to GitHub → `app.py` → Edit (pencil icon)
2. **Select all** and **delete** the current content
3. **Paste** the new code above
4. Click **"Commit changes"**
5. Wait 1-2 minutes for Streamlit Cloud to redeploy

---

Also make sure your `requirements.txt` is the minimal version:
```
streamlit
pandas
numpy
plotly
scikit-learn
lightgbm
shap
joblib
openai
anthropic
google-generativeai
Pillow
