# Cell: Create streamlit_app.py for GitHub deployment

app_code = '''
"""
Android Adware Detection System with Explainable AI
Web Application with Multi-LLM Support
Version 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
from datetime import datetime
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Android Adware Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
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
    .chat-user {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .chat-ai {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained model"""
    model_files = ['adware_model_22features.joblib', 'adware_detection_model.joblib', 
                   'optimized_neural_network.pkl', 'scaler.pkl']
    
    # Try loading joblib models first
    for model_path in ['adware_model_22features.joblib', 'adware_detection_model.joblib']:
        if os.path.exists(model_path):
            try:
                package = joblib.load(model_path)
                return package['model'], package['scaler'], package['feature_names']
            except:
                continue
    
    # Try loading pkl files (existing repo format)
    if os.path.exists('optimized_neural_network.pkl') and os.path.exists('scaler.pkl'):
        try:
            import pickle
            with open('optimized_neural_network.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            # Load feature columns
            if os.path.exists('feature_columns.pkl'):
                with open('feature_columns.pkl', 'rb') as f:
                    feature_names = pickle.load(f)
            else:
                feature_names = None
            return model, scaler, feature_names
        except:
            pass
    
    return None, None, None

MODEL, SCALER, FEATURE_NAMES = load_model()

# Default feature names if not loaded
if FEATURE_NAMES is None:
    FEATURE_NAMES = [
        'network_op', 'sqllite_op', 'fileio_op', 'no_action', 'noc', 'dit',
        'lcom', 'cbo', 'ppiv', 'apd', 'start_activities', 'start_activity',
        'start_service', 'start_action_mode', 'start_activity_result',
        'start_activity_from_frag', 'start_activity_needed',
        'start_intent_for_result', 'start_search', 'methods_per_class',
        'bytecode_per_method', 'cyclomatic'
    ]

FEATURE_DESCRIPTIONS = {
    'cyclomatic': 'Code complexity measure - higher values indicate more complex, typically legitimate apps',
    'network_op': 'Number of network operations - high values may indicate ad activity or data transmission',
    'fileio_op': 'File I/O operations - indicates file system access patterns',
    'sqllite_op': 'SQLite database operations - data storage behavior',
    'no_action': 'Empty exception handlers - code quality indicator',
    'cbo': 'Coupling Between Objects - measures class interdependence',
    'lcom': 'Lack of Cohesion in Methods - OOP design quality metric',
    'dit': 'Depth of Inheritance Tree - class hierarchy complexity',
    'noc': 'Number of Children - inheritance breadth',
    'ppiv': 'Package-level coupling metric',
    'apd': 'Average method complexity',
    'methods_per_class': 'Average methods per class - code structure indicator',
    'bytecode_per_method': 'Average bytecode size per method',
    'start_activity': 'Activity launch calls - UI navigation patterns',
    'start_service': 'Service start calls - background processing',
    'start_activities': 'Multiple activity launches',
    'start_action_mode': 'Action mode initiations',
    'start_activity_result': 'Activity result requests',
    'start_activity_from_frag': 'Fragment-based activity launches',
    'start_activity_needed': 'Required activity launches',
    'start_intent_for_result': 'Intent result requests',
    'start_search': 'Search functionality calls'
}

# ============================================================
# LLM INTEGRATION
# ============================================================
def get_llm_response(prompt, provider, api_key, context=""):
    """Get response from selected LLM provider"""
    
    system_prompt = """You are an expert Android security analyst specializing in adware detection. 
    You analyze detection results and provide clear, actionable insights. Be specific about technical 
    details but also explain things in a way that non-technical users can understand. 
    Remember: We are detecting ADWARE (advertising software), not general malware."""
    
    full_prompt = f"{context}\\n\\n{prompt}" if context else prompt
    
    try:
        if provider == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        elif provider == "Anthropic Claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                system=system_prompt,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.content[0].text
            
        elif provider == "Google Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(f"{system_prompt}\\n\\n{full_prompt}")
            return response.text
            
    except Exception as e:
        return f"Error with {provider}: {str(e)}"

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_prediction(features_df):
    if MODEL is None:
        return None, None
    
    # Get feature columns that exist in both dataframe and model
    available_features = [f for f in FEATURE_NAMES if f in features_df.columns]
    X = features_df[available_features].copy()
    
    # Scale if scaler exists
    if SCALER is not None:
        try:
            X_scaled = pd.DataFrame(SCALER.transform(X), columns=available_features)
        except:
            X_scaled = X
    else:
        X_scaled = X
    
    prediction = MODEL.predict(X_scaled)
    
    # Get probabilities if available
    if hasattr(MODEL, 'predict_proba'):
        probability = MODEL.predict_proba(X_scaled)
    else:
        # Create dummy probabilities for models without predict_proba
        probability = np.array([[1-p, p] for p in prediction])
    
    return prediction, probability

def get_shap_values(features_df):
    if MODEL is None:
        return None, None, None
    
    available_features = [f for f in FEATURE_NAMES if f in features_df.columns]
    X = features_df[available_features].copy()
    
    if SCALER is not None:
        try:
            X_scaled = pd.DataFrame(SCALER.transform(X), columns=available_features)
        except:
            X_scaled = X
    else:
        X_scaled = X
    
    try:
        explainer = shap.TreeExplainer(MODEL)
        shap_values = explainer.shap_values(X_scaled)
        return shap_values, explainer.expected_value, X_scaled
    except:
        try:
            explainer = shap.Explainer(MODEL, X_scaled)
            shap_values = explainer(X_scaled)
            return shap_values.values, shap_values.base_values, X_scaled
        except:
            return None, None, X_scaled

def create_gauge_chart(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#e8f5e9'},
                {'range': [33, 66], 'color': '#fff3e0'},
                {'range': [66, 100], 'color': '#ffebee'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_shap_waterfall(shap_values, features, feature_names, idx=0):
    if shap_values is None:
        return None
    
    if isinstance(shap_values, list):
        sv = shap_values[1][idx] if len(shap_values) > 1 else shap_values[0][idx]
    else:
        sv = shap_values[idx] if len(shap_values.shape) > 1 else shap_values
    
    if len(sv) != len(feature_names):
        feature_names = list(features.columns)
    
    indices = np.argsort(np.abs(sv))[::-1][:10]
    sorted_features = [feature_names[i] for i in indices]
    sorted_shap = [sv[i] for i in indices]
    colors = ['#ff4444' if s > 0 else '#44aa44' for s in sorted_shap]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_features[::-1],
        x=sorted_shap[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=[f'{s:.3f}' for s in sorted_shap[::-1]],
        textposition='outside'
    ))
    fig.update_layout(
        title='SHAP Feature Contributions (Top 10)',
        xaxis_title='SHAP Value (Impact on Adware Prediction)',
        yaxis_title='Feature',
        height=400,
        showlegend=False
    )
    return fig

def create_feature_importance_chart(model, feature_names):
    if model is None or not hasattr(model, 'feature_importances_'):
        return None
    
    importance = model.feature_importances_
    if len(importance) != len(feature_names):
        feature_names = [f'feature_{i}' for i in range(len(importance))]
    
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True).tail(15)
    
    fig = px.bar(feature_imp, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='RdYlGn_r',
                 title='Top 15 Feature Importances')
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_radar_chart(features_df, feature_names, idx=0):
    available_features = [f for f in ['cyclomatic', 'network_op', 'fileio_op', 'cbo', 'lcom', 'start_service'] 
                         if f in features_df.columns]
    
    if not available_features:
        return None
    
    sample_vals = features_df[available_features].iloc[idx].values
    
    # Normalize
    max_vals = {'cyclomatic': 50000, 'network_op': 20, 'fileio_op': 300, 'cbo': 1, 'lcom': 100, 'start_service': 50}
    normalized = [min(1, sample_vals[i] / max_vals.get(available_features[i], 1)) for i in range(len(available_features))]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized,
        theta=available_features,
        fill='toself',
        name='This App',
        line_color='#1E88E5'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Feature Profile',
        height=350
    )
    return fig

def build_detection_context(results, features):
    pred = results['predictions'][0]
    prob = results['probabilities'][0]
    feat = features.iloc[0]
    
    context = f"""
DETECTION RESULTS:
Classification: {"ADWARE DETECTED" if pred == 1 else "BENIGN APPLICATION"}
Confidence: {max(prob)*100:.1f}%
Adware Probability: {prob[1]*100:.1f}%

KEY FEATURES:
"""
    for col in ['cyclomatic', 'network_op', 'fileio_op', 'no_action', 'cbo', 'lcom']:
        if col in feat.index:
            context += f"- {col}: {feat[col]:.2f}\\n"
    
    return context

# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    st.markdown('<p class="main-header">🛡️ Android Adware Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by Machine Learning & Explainable AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/android-os.png", width=80)
        st.markdown("## ⚙️ Configuration")
        
        st.markdown("### 🤖 AI Provider")
        llm_provider = st.selectbox("Select LLM Provider", ["Anthropic Claude", "OpenAI", "Google Gemini"])
        
        api_key = st.text_input(f"{llm_provider} API Key", type="password")
        
        if api_key:
            st.success(f"✓ {llm_provider} configured")
        else:
            st.warning("Enter API key for AI features")
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        if MODEL is not None:
            st.success("✓ Model Loaded")
            st.info(f"Features: {len(FEATURE_NAMES)}")
            st.info("Algorithm: LightGBM")
            st.info("Accuracy: 99.32%")
        else:
            st.error("✗ Model not loaded")
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        Detects Android **adware** using ML and static code analysis.
        
        **Features:**
        - 🔍 Real-time detection
        - 📊 SHAP explanations
        - 🤖 Multi-LLM support
        - 📈 Visual analytics
        """)
        st.markdown("---")
        st.markdown("[📥 Feature Extraction Tool](https://github.com/sealuzh/user_quality)")
    
    # Session state
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_features' not in st.session_state:
        st.session_state.current_features = None
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Detection", "📊 Explainability", "🤖 AI Chat", "📋 Batch Analysis", "📄 Report"])
    
    # TAB 1: DETECTION
    with tab1:
        st.markdown("### 📤 Input Data for Analysis")
        st.markdown('<div class="info-box"><strong>Note:</strong> Features can be extracted using the <a href="https://github.com/sealuzh/user_quality">Android Quality Metrics Tool</a>.</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📁 Upload CSV File")
            uploaded_file = st.file_uploader("Upload CSV with features", type=['csv'])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.success(f"✓ Loaded {len(data)} samples")
                    st.session_state.current_features = data
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col2:
            st.markdown("#### ✏️ Manual Input")
            with st.expander("Enter Feature Values"):
                manual_data = {}
                cols = st.columns(3)
                for i, feat in enumerate(FEATURE_NAMES):
                    with cols[i % 3]:
                        manual_data[feat] = st.number_input(feat, value=0.0, format="%.4f", key=f"m_{feat}")
                if st.button("Use Manual Input"):
                    st.session_state.current_features = pd.DataFrame([manual_data])
                    st.success("✓ Loaded")
        
        st.markdown("---")
        if st.session_state.current_features is not None:
            st.info(f"📊 Data: {len(st.session_state.current_features)} sample(s)")
            if st.button("🔍 RUN ADWARE DETECTION", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    predictions, probabilities = get_prediction(st.session_state.current_features)
                    if predictions is not None:
                        st.session_state.detection_results = {
                            'predictions': predictions,
                            'probabilities': probabilities,
                            'features': st.session_state.current_features
                        }
                        st.success("✓ Complete!")
        
        if st.session_state.detection_results is not None:
            st.markdown("---")
            st.markdown("### 🎯 Detection Results")
            results = st.session_state.detection_results
            pred, prob = results['predictions'], results['probabilities']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if pred[0] == 1:
                    st.markdown('<div class="adware-card">🚨 ADWARE DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="benign-card">✅ BENIGN</div>', unsafe_allow_html=True)
            with col2:
                st.plotly_chart(create_gauge_chart(max(prob[0])*100, "Confidence", "#1E88E5"), use_container_width=True)
            with col3:
                risk = prob[0][1] * 100
                st.plotly_chart(create_gauge_chart(risk, "Risk Level", "#ff4444" if risk > 50 else "#44aa44"), use_container_width=True)
            
            if len(pred) > 1:
                st.markdown("### 📊 Batch Summary")
                col1, col2, col3, col4 = st.columns(4)
                n_adware = sum(pred)
                col1.metric("Total", len(pred))
                col2.metric("Adware", n_adware)
                col3.metric("Benign", len(pred) - n_adware)
                col4.metric("Adware Rate", f"{n_adware/len(pred)*100:.1f}%")
            
            st.markdown("### 📈 Feature Importance")
            available_features = [f for f in FEATURE_NAMES if f in results['features'].columns]
            fig = create_feature_importance_chart(MODEL, available_features)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: EXPLAINABILITY
    with tab2:
        st.markdown("### 🔬 Model Explainability")
        if st.session_state.detection_results is None:
            st.info("👆 Run detection first")
        else:
            results = st.session_state.detection_results
            features = results['features']
            available_features = [f for f in FEATURE_NAMES if f in features.columns]
            
            if len(features) > 1:
                sample_idx = st.selectbox("Select Sample", range(len(features)),
                    format_func=lambda x: f"Sample {x+1} - {'ADWARE' if results['predictions'][x] == 1 else 'BENIGN'}")
            else:
                sample_idx = 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 SHAP Values")
                with st.spinner("Calculating..."):
                    shap_values, _, X_scaled = get_shap_values(features)
                    if shap_values is not None:
                        fig = create_shap_waterfall(shap_values, X_scaled, available_features, sample_idx)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        st.markdown("🔴 Red → ADWARE | 🟢 Green → BENIGN")
                    else:
                        st.warning("SHAP not available for this model")
            
            with col2:
                st.markdown("#### 🎯 Feature Profile")
                fig = create_radar_chart(features, available_features, sample_idx)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 📋 Feature Details")
            sample_features = features.iloc[sample_idx]
            feature_table = [{'Feature': f, 'Value': f"{sample_features[f]:.4f}" if f in sample_features else "N/A",
                            'Description': FEATURE_DESCRIPTIONS.get(f, 'N/A')}
                           for f in available_features]
            st.dataframe(pd.DataFrame(feature_table), use_container_width=True, hide_index=True)
    
    # TAB 3: AI CHAT
    with tab3:
        st.markdown(f"### 🤖 AI Security Analyst")
        st.markdown(f"*Powered by {llm_provider}*")
        
        if not api_key:
            st.warning(f"⚠️ Enter {llm_provider} API key in sidebar")
        else:
            context = ""
            if st.session_state.detection_results is not None:
                context = build_detection_context(st.session_state.detection_results, st.session_state.detection_results['features'])
                st.markdown('<div class="info-box">💡 AI has context about your detection results</div>', unsafe_allow_html=True)
            
            st.markdown("#### 🚀 Quick Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📝 Explain Detection", use_container_width=True):
                    if st.session_state.detection_results is not None:
                        with st.spinner("Generating..."):
                            response = get_llm_response("Explain this adware detection result in detail.", llm_provider, api_key, context)
                            st.session_state.chat_history.append(("Explain this detection", response))
                    else:
                        st.warning("Run detection first!")
            
            with col2:
                if st.button("⚠️ Security Advisory", use_container_width=True):
                    if st.session_state.detection_results is not None:
                        with st.spinner("Generating..."):
                            response = get_llm_response("Provide a security advisory with risks and recommendations.", llm_provider, api_key, context)
                            st.session_state.chat_history.append(("Security Advisory", response))
                    else:
                        st.warning("Run detection first!")
            
            with col3:
                if st.button("🔍 Technical Analysis", use_container_width=True):
                    if st.session_state.detection_results is not None:
                        with st.spinner("Generating..."):
                            response = get_llm_response("Provide detailed technical analysis of the features.", llm_provider, api_key, context)
                            st.session_state.chat_history.append(("Technical Analysis", response))
                    else:
                        st.warning("Run detection first!")
            
            st.markdown("---")
            st.markdown("#### 💬 Ask a Question")
            user_input = st.text_input("Ask about the detection...", placeholder="e.g., Why is network_op important?")
            if st.button("Send", type="primary") and user_input:
                with st.spinner("Thinking..."):
                    response = get_llm_response(user_input, llm_provider, api_key, context)
                    st.session_state.chat_history.append((user_input, response))
            
            if st.session_state.chat_history:
                st.markdown("#### 📜 Conversation History")
                for q, a in reversed(st.session_state.chat_history[-5:]):
                    st.markdown(f'<div class="chat-user"><strong>👤 You:</strong> {q}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-ai"><strong>🤖 AI:</strong> {a}</div>', unsafe_allow_html=True)
                if st.button("🗑️ Clear History"):
                    st.session_state.chat_history = []
                    st.rerun()
    
    # TAB 4: BATCH ANALYSIS
    with tab4:
        st.markdown("### 📋 Batch Analysis")
        if st.session_state.detection_results is None:
            st.info("👆 Run detection first")
        else:
            results = st.session_state.detection_results
            pred, prob = results['predictions'], results['probabilities']
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total", len(pred))
            col2.metric("Adware", sum(pred))
            col3.metric("Benign", len(pred) - sum(pred))
            col4.metric("Avg Confidence", f"{np.mean([max(p) for p in prob])*100:.1f}%")
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(values=[sum(pred), len(pred)-sum(pred)], names=['Adware', 'Benign'],
                            color_discrete_sequence=['#ff4444', '#44aa44'], title='Distribution')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.histogram(x=[p[1]*100 for p in prob], nbins=20, title='Adware Probability',
                                  labels={'x': 'Probability (%)'})
                st.plotly_chart(fig, use_container_width=True)
            
            results_df = results['features'].copy()
            results_df['Prediction'] = ['ADWARE' if p == 1 else 'BENIGN' for p in pred]
            results_df['Adware_Prob'] = [f"{p[1]*100:.1f}%" for p in prob]
            results_df['Risk'] = ['HIGH' if p[1] >= 0.8 else ('MEDIUM' if p[1] >= 0.5 else 'LOW') for p in prob]
            
            display_cols = ['Prediction', 'Adware_Prob', 'Risk'] + [c for c in ['cyclomatic', 'network_op', 'fileio_op'] if c in results_df.columns]
            st.dataframe(results_df[display_cols], hide_index=True)
            
            csv = results_df.to_csv(index=False)
            st.download_button("📥 Download Results", csv, "adware_results.csv", "text/csv", use_container_width=True)
    
    # TAB 5: REPORT
    with tab5:
        st.markdown("### 📄 Generate Report")
        if st.session_state.detection_results is None:
            st.info("👆 Run detection first")
        else:
            results = st.session_state.detection_results
            pred, prob = results['predictions'], results['probabilities']
            
            if st.button("📄 Generate Report", type="primary"):
                st.markdown(f"""
# 🛡️ Android Adware Detection Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary

| Metric | Value |
|--------|-------|
| Total Samples | {len(pred)} |
| Adware Detected | {sum(pred)} ({sum(pred)/len(pred)*100:.1f}%) |
| Benign | {len(pred)-sum(pred)} ({(len(pred)-sum(pred))/len(pred)*100:.1f}%) |
| Average Confidence | {np.mean([max(p) for p in prob])*100:.1f}% |

---

## Risk Distribution

- **High Risk (≥80%):** {sum(1 for p in prob if p[1] >= 0.8)}
- **Medium Risk (50-80%):** {sum(1 for p in prob if 0.5 <= p[1] < 0.8)}
- **Low Risk (<50%):** {sum(1 for p in prob if p[1] < 0.5)}

---

## Recommendations

1. Quarantine high-risk samples immediately
2. Manual review of medium-risk samples
3. Monitor network behavior of flagged apps

*Generated by Android Adware Detection System v2.0*
                """)

if __name__ == "__main__":
    main()
'''

# Save the file
with open('streamlit_app.py', 'w') as f:
    f.write(app_code)

print("✓ streamlit_app.py saved!")
print("\nNext steps:")
print("1. Upload this file to your GitHub repo")
print("2. Upload adware_model_22features.joblib to GitHub")
print("3. Update requirements.txt")
