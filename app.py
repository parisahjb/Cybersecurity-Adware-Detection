"""
Android Adware Detection System with Explainable AI
Version 2.1 - Complete with Radar Charts, Counterfactuals, Manual Input
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
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
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
    'noc': 'Number of Children - inheritance breadth',
    'start_service': 'Service start calls - background processing',
    'start_activity': 'Activity launch calls - UI navigation',
    'methods_per_class': 'Average methods per class',
    'bytecode_per_method': 'Average bytecode per method',
    'ppiv': 'Package-level coupling metric',
    'apd': 'Average method complexity'
}

# Typical values for reference (from our analysis)
TYPICAL_VALUES = {
    'benign': {
        'cyclomatic': 38606, 'network_op': 0.26, 'fileio_op': 163,
        'no_action': 831, 'cbo': 0.58, 'lcom': 32.5, 'start_service': 5.2
    },
    'adware': {
        'cyclomatic': 5754, 'network_op': 3.89, 'fileio_op': 54,
        'no_action': 183, 'cbo': 0.52, 'lcom': 28.1, 'start_service': 3.1
    }
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

def create_radar_chart(features_df, idx=0):
    """Create radar chart for feature profile"""
    key_features = ['cyclomatic', 'network_op', 'fileio_op', 'cbo', 'lcom', 'start_service']
    available = [f for f in key_features if f in features_df.columns]
    
    if not available:
        return None
    
    sample_vals = features_df[available].iloc[idx].values
    
    # Normalize values (0-1 scale)
    max_vals = {'cyclomatic': 50000, 'network_op': 20, 'fileio_op': 300, 
                'cbo': 1, 'lcom': 100, 'start_service': 50}
    normalized = [min(1, abs(sample_vals[i]) / max_vals.get(available[i], 1)) for i in range(len(available))]
    
    # Close the radar chart
    normalized.append(normalized[0])
    available_plot = available + [available[0]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized,
        theta=available_plot,
        fill='toself',
        name='This App',
        line_color='#1E88E5',
        fillcolor='rgba(30, 136, 229, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title='Feature Profile (Normalized)',
        height=350
    )
    return fig

def get_counterfactual_suggestions(features_df, pred, idx=0):
    """Generate counterfactual explanations"""
    suggestions = []
    sample = features_df.iloc[idx]
    
    if pred[idx] == 1:  # Classified as adware
        # Suggest changes to appear benign
        if 'cyclomatic' in sample and sample['cyclomatic'] < 15000:
            suggestions.append({
                'feature': 'cyclomatic',
                'current': f"{sample['cyclomatic']:.0f}",
                'suggested': '> 15,000',
                'reason': 'Increase code complexity (benign avg: 38,606)'
            })
        if 'network_op' in sample and sample['network_op'] > 2:
            suggestions.append({
                'feature': 'network_op',
                'current': f"{sample['network_op']:.1f}",
                'suggested': '< 2',
                'reason': 'Reduce network operations (benign avg: 0.26)'
            })
        if 'fileio_op' in sample and sample['fileio_op'] < 100:
            suggestions.append({
                'feature': 'fileio_op',
                'current': f"{sample['fileio_op']:.0f}",
                'suggested': '> 100',
                'reason': 'Increase file I/O operations (benign avg: 163)'
            })
        if 'no_action' in sample and sample['no_action'] < 400:
            suggestions.append({
                'feature': 'no_action',
                'current': f"{sample['no_action']:.0f}",
                'suggested': '> 400',
                'reason': 'More exception handlers (benign avg: 831)'
            })
    else:  # Classified as benign
        suggestions.append({
            'feature': 'N/A',
            'current': 'N/A',
            'suggested': 'N/A',
            'reason': 'App classified as benign - no changes needed'
        })
    
    return suggestions

# LLM function
def get_llm_response(prompt, provider, api_key, context=""):
    system = "You are an Android security analyst specializing in adware detection. Provide clear, actionable insights."
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
        st.markdown("## Configuration")
        
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
    
    # ============================================
    # TAB 1: Detection
    # ============================================
    with tab1:
        st.markdown("### Input Data")
        st.markdown('<div class="info-box"><strong>Note:</strong> Features can be extracted using the <a href="https://github.com/sealuzh/user_quality">Android Quality Metrics Tool</a>.</div>', unsafe_allow_html=True)
        
        input_method = st.radio("Select Input Method:", ["Upload CSV", "Load Sample Data", "Manual Input"], horizontal=True)
        
        if input_method == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV with features", type=['csv'])
            if uploaded:
                st.session_state.features = pd.read_csv(uploaded)
                st.success(f"✓ Loaded {len(st.session_state.features)} samples")
        
        elif input_method == "Load Sample Data":
            if st.button("📂 Load Sample Data"):
                if os.path.exists('sample_test_data.csv'):
                    st.session_state.features = pd.read_csv('sample_test_data.csv')
                    st.success("✓ Sample data loaded")
                else:
                    st.error("Sample file not found")
        
        elif input_method == "Manual Input":
            st.markdown("#### Enter Feature Values")
            st.markdown("*Enter values for each feature (use 0 if unknown)*")
            
            manual_data = {}
            
            # Group features for better organization
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Code Metrics**")
                manual_data['cyclomatic'] = st.number_input('cyclomatic', value=5000.0, format="%.1f", help="Code complexity measure")
                manual_data['methods_per_class'] = st.number_input('methods_per_class', value=5.0, format="%.2f")
                manual_data['bytecode_per_method'] = st.number_input('bytecode_per_method', value=20.0, format="%.2f")
                manual_data['lcom'] = st.number_input('lcom', value=30.0, format="%.2f", help="Lack of Cohesion")
                manual_data['cbo'] = st.number_input('cbo', value=0.5, format="%.3f", help="Coupling Between Objects")
                manual_data['dit'] = st.number_input('dit', value=2.0, format="%.2f", help="Depth of Inheritance")
                manual_data['noc'] = st.number_input('noc', value=1.0, format="%.2f", help="Number of Children")
                manual_data['ppiv'] = st.number_input('ppiv', value=0.0, format="%.2f")
            
            with col2:
                st.markdown("**Behavioral Features**")
                manual_data['network_op'] = st.number_input('network_op', value=4.0, format="%.1f", help="Network operations count")
                manual_data['fileio_op'] = st.number_input('fileio_op', value=50.0, format="%.1f", help="File I/O operations")
                manual_data['sqllite_op'] = st.number_input('sqllite_op', value=0.0, format="%.1f", help="SQLite operations")
                manual_data['no_action'] = st.number_input('no_action', value=200.0, format="%.1f", help="Empty exception handlers")
                manual_data['apd'] = st.number_input('apd', value=0.0, format="%.2f")
            
            with col3:
                st.markdown("**API Calls**")
                manual_data['start_activity'] = st.number_input('start_activity', value=5.0, format="%.1f")
                manual_data['start_activities'] = st.number_input('start_activities', value=0.0, format="%.1f")
                manual_data['start_service'] = st.number_input('start_service', value=3.0, format="%.1f")
                manual_data['start_action_mode'] = st.number_input('start_action_mode', value=0.0, format="%.1f")
                manual_data['start_activity_result'] = st.number_input('start_activity_result', value=0.0, format="%.1f")
                manual_data['start_activity_from_frag'] = st.number_input('start_activity_from_frag', value=0.0, format="%.1f")
                manual_data['start_activity_needed'] = st.number_input('start_activity_needed', value=0.0, format="%.1f")
                manual_data['start_intent_for_result'] = st.number_input('start_intent_for_result', value=0.0, format="%.1f")
                manual_data['start_search'] = st.number_input('start_search', value=0.0, format="%.1f")
            
            if st.button("✅ Use Manual Input", type="primary"):
                st.session_state.features = pd.DataFrame([manual_data])
                st.success("✓ Manual input loaded")
        
        # Show current data status
        st.markdown("---")
        if st.session_state.features is not None:
            st.info(f"📊 {len(st.session_state.features)} sample(s) ready for analysis")
            
            with st.expander("Preview Data"):
                st.dataframe(st.session_state.features.head(), use_container_width=True)
            
            if st.button("🔍 RUN ADWARE DETECTION", type="primary", use_container_width=True):
                pred, prob = get_prediction(st.session_state.features)
                if pred is not None:
                    st.session_state.results = {'pred': pred, 'prob': prob}
                    st.success("✓ Detection complete!")
        
        # Results
        if st.session_state.results is not None:
            st.markdown("---")
            st.markdown("### Detection Results")
            
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
                st.plotly_chart(create_gauge(prob[0][1]*100, "Risk Level", "#ff4444" if prob[0][1] > 0.5 else "#44aa44"), use_container_width=True)
            
            # Batch summary
            if len(pred) > 1:
                st.markdown("### Batch Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total", len(pred))
                col2.metric("Adware", sum(pred))
                col3.metric("Benign", len(pred) - sum(pred))
                col4.metric("Adware Rate", f"{sum(pred)/len(pred)*100:.1f}%")
            
            # Feature importance
            if hasattr(MODEL, 'feature_importances_'):
                st.markdown("### Feature Importance")
                available = [f for f in FEATURE_NAMES if f in st.session_state.features.columns]
                imp_values = MODEL.feature_importances_[:len(available)]
                imp = pd.DataFrame({'Feature': available[:len(imp_values)], 
                                   'Importance': imp_values}).sort_values('Importance', ascending=True).tail(15)
                fig = px.bar(imp, x='Importance', y='Feature', orientation='h', 
                            color='Importance', color_continuous_scale='RdYlGn_r')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # TAB 2: Explainability
    # ============================================
    with tab2:
        st.markdown("### Model Explainability")
        
        if st.session_state.results is None:
            st.info("👆 Run detection first to see explanations")
        else:
            pred = st.session_state.results['pred']
            prob = st.session_state.results['prob']
            features = st.session_state.features
            
            # Sample selector
            if len(features) > 1:
                sample_idx = st.selectbox(
                    "Select Sample to Analyze",
                    range(len(features)),
                    format_func=lambda x: f"Sample {x+1} - {'ADWARE' if pred[x] == 1 else 'BENIGN'} ({prob[x][1]*100:.1f}%)"
                )
            else:
                sample_idx = 0
            
            col1, col2 = st.columns(2)
            
            # SHAP Values
            with col1:
                st.markdown("#### SHAP Feature Contributions")
                try:
                    import shap
                    available = [f for f in FEATURE_NAMES if f in features.columns]
                    X = features[available]
                    if SCALER:
                        X_scaled = pd.DataFrame(SCALER.transform(X), columns=available)
                    else:
                        X_scaled = X
                    
                    explainer = shap.TreeExplainer(MODEL)
                    shap_vals = explainer.shap_values(X_scaled)
                    
                    if isinstance(shap_vals, list):
                        sv = shap_vals[1][sample_idx]
                    else:
                        sv = shap_vals[sample_idx]
                    
                    idx = np.argsort(np.abs(sv))[::-1][:10]
                    fig = go.Figure(go.Bar(
                        y=[available[i] for i in idx][::-1],
                        x=[sv[i] for i in idx][::-1],
                        orientation='h',
                        marker_color=['#ff4444' if s > 0 else '#44aa44' for s in [sv[i] for i in idx][::-1]]
                    ))
                    fig.update_layout(title='SHAP Values (Top 10)', height=400,
                                     xaxis_title='SHAP Value (Impact on Adware Prediction)')
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("🔴 Red → Pushes toward **ADWARE** | 🟢 Green → Pushes toward **BENIGN**")
                except Exception as e:
                    st.warning(f"SHAP calculation unavailable: {e}")
            
            # Radar Chart
            with col2:
                st.markdown("#### Feature Profile")
                radar_fig = create_radar_chart(features, sample_idx)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True)
                else:
                    st.warning("Radar chart unavailable")
            
            # Counterfactual Analysis
            st.markdown("---")
            st.markdown("#### Counterfactual Analysis")
            st.markdown("*What changes would alter the classification?*")
            
            suggestions = get_counterfactual_suggestions(features, pred, sample_idx)
            
            if suggestions:
                cf_df = pd.DataFrame(suggestions)
                st.dataframe(cf_df, use_container_width=True, hide_index=True)
            
            # Feature Details Table
            st.markdown("---")
            st.markdown("#### Feature Details")
            sample = features.iloc[sample_idx]
            table_data = []
            for f in FEATURE_NAMES:
                if f in sample:
                    table_data.append({
                        'Feature': f,
                        'Value': f"{sample[f]:.4f}",
                        'Description': FEATURE_DESCRIPTIONS.get(f, '')
                    })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    
    # ============================================
    # TAB 3: AI Chat
    # ============================================
    with tab3:
        st.markdown(f"### 🤖 AI Security Analyst ({llm_provider})")
        
        if not api_key:
            st.warning("⚠️ Enter API key in sidebar to use AI features")
        else:
            context = ""
            if st.session_state.results:
                pred = st.session_state.results['pred'][0]
                prob = st.session_state.results['prob'][0]
                feat = st.session_state.features.iloc[0]
                context = f"""Detection Result: {'ADWARE' if pred == 1 else 'BENIGN'}
Confidence: {max(prob)*100:.1f}%
Key Features: cyclomatic={feat.get('cyclomatic', 'N/A')}, network_op={feat.get('network_op', 'N/A')}, fileio_op={feat.get('fileio_op', 'N/A')}"""
                st.markdown('<div class="info-box">💡 AI has context about your current detection results</div>', unsafe_allow_html=True)
            
            st.markdown("#### Quick Actions")
            col1, col2, col3 = st.columns(3)
            
            if col1.button("📝 Explain Detection", use_container_width=True):
                if st.session_state.results:
                    with st.spinner("Analyzing..."): 
                        r = get_llm_response("Explain this adware detection result in detail. What are the key indicators?", llm_provider, api_key, context)
                        st.session_state.chat.append(("Explain Detection", r))
                else:
                    st.warning("Run detection first!")
            
            if col2.button("⚠️ Security Advisory", use_container_width=True):
                if st.session_state.results:
                    with st.spinner("Generating advisory..."): 
                        r = get_llm_response("Provide a security advisory. What risks does this pose? What actions should users take?", llm_provider, api_key, context)
                        st.session_state.chat.append(("Security Advisory", r))
                else:
                    st.warning("Run detection first!")
            
            if col3.button("🔍 Technical Analysis", use_container_width=True):
                if st.session_state.results:
                    with st.spinner("Analyzing..."): 
                        r = get_llm_response("Provide detailed technical analysis of the feature values and what they indicate about app behavior.", llm_provider, api_key, context)
                        st.session_state.chat.append(("Technical Analysis", r))
                else:
                    st.warning("Run detection first!")
            
            st.markdown("---")
            st.markdown("#### Ask a Question")
            q = st.text_input("Ask about the detection...", placeholder="e.g., Why is network_op important?")
            if st.button("Send", type="primary") and q:
                with st.spinner("Thinking..."): 
                    r = get_llm_response(q, llm_provider, api_key, context)
                    st.session_state.chat.append((q, r))
            
            # Chat history
            if st.session_state.chat:
                st.markdown("---")
                st.markdown("#### Conversation History")
                for question, answer in reversed(st.session_state.chat[-5:]):
                    st.markdown(f"**👤 You:** {question}")
                    st.markdown(f"**🤖 AI:** {answer}")
                    st.markdown("---")
                
                if st.button("🗑️ Clear History"):
                    st.session_state.chat = []
                    st.rerun()
    
    # ============================================
    # TAB 4: Batch Analysis
    # ============================================
    with tab4:
        st.markdown("### Batch Analysis")
        
        if st.session_state.results is None:
            st.info("👆 Run detection first to see batch results")
        else:
            pred = st.session_state.results['pred']
            prob = st.session_state.results['prob']
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Samples", len(pred))
            col2.metric("Adware Detected", sum(pred))
            col3.metric("Benign", len(pred) - sum(pred))
            col4.metric("Avg Confidence", f"{np.mean([max(p) for p in prob])*100:.1f}%")
            
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(values=[sum(pred), len(pred)-sum(pred)], 
                            names=['Adware', 'Benign'],
                            color_discrete_sequence=['#ff4444', '#44aa44'],
                            title='Detection Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(x=[p[1]*100 for p in prob], nbins=20, 
                                  title='Adware Probability Distribution',
                                  labels={'x': 'Probability (%)'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.markdown("### Detailed Results")
            df = st.session_state.features.copy()
            df['Prediction'] = ['ADWARE' if p == 1 else 'BENIGN' for p in pred]
            df['Adware_Prob'] = [f"{p[1]*100:.1f}%" for p in prob]
            df['Risk'] = ['HIGH' if p[1] >= 0.8 else ('MEDIUM' if p[1] >= 0.5 else 'LOW') for p in prob]
            
            display_cols = ['Prediction', 'Adware_Prob', 'Risk'] + [c for c in ['cyclomatic', 'network_op', 'fileio_op'] if c in df.columns]
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results (CSV)", csv, "adware_detection_results.csv", "text/csv", use_container_width=True)
    
    # ============================================
    # TAB 5: Report
    # ============================================
    with tab5:
        st.markdown("### Generate Report")
        
        if st.session_state.results is None:
            st.info("👆 Run detection first to generate a report")
        else:
            report_type = st.selectbox("Report Type", ["Executive Summary", "Technical Report", "Full Analysis"])
            
            if st.button("📄 Generate Report", type="primary"):
                pred = st.session_state.results['pred']
                prob = st.session_state.results['prob']
                
                st.markdown("---")
                st.markdown(f"""
# 🛡️ Android Adware Detection Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Report Type:** {report_type}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Samples Analyzed | {len(pred)} |
| Adware Detected | {sum(pred)} ({sum(pred)/len(pred)*100:.1f}%) |
| Benign Applications | {len(pred)-sum(pred)} ({(len(pred)-sum(pred))/len(pred)*100:.1f}%) |
| Average Confidence | {np.mean([max(p) for p in prob])*100:.1f}% |
| High Risk Samples | {sum(1 for p in prob if p[1] >= 0.8)} |

---

## Risk Distribution

- **High Risk (≥80%):** {sum(1 for p in prob if p[1] >= 0.8)} samples
- **Medium Risk (50-80%):** {sum(1 for p in prob if 0.5 <= p[1] < 0.8)} samples
- **Low Risk (<50%):** {sum(1 for p in prob if p[1] < 0.5)} samples

---

## Recommendations

1. **Immediate Action:** Quarantine all high-risk applications
2. **Review Required:** Manually analyze medium-risk applications
3. **Monitoring:** Enable network monitoring for flagged applications
4. **User Advisory:** Notify users about detected adware threats

---

## Model Information

- **Algorithm:** LightGBM (Gradient Boosting)
- **Accuracy:** 99.32%
- **Features Used:** {len(FEATURE_NAMES)}

---

*Report generated by Android Adware Detection System v2.1*
                """)

if __name__ == "__main__":
    main()
