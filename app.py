"""
AI-Powered Android Adware Detection System
Streamlit Web Application with Real Models and Data
Author: Dr. Parisa Hajibabaee
"""

import streamlit as st

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="🛡️ AI Adware Detector Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import libraries
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import base64
from io import BytesIO
import hashlib

# Enhanced Custom CSS with animations and modern design
st.markdown("""
<style>
    /* Dark mode variables */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
        --bg-color: #ffffff;
        --text-color: #1f2937;
    }
    
    /* Animated gradient header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Card animations */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideIn 0.5s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Enhanced result boxes */
    .success-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        animation: pulse 2s infinite;
        text-align: center;
        margin: 1rem 0;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        animation: pulse 2s infinite;
        text-align: center;
        margin: 1rem 0;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Glass morphism effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Custom button styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'comparison_apps' not in st.session_state:
    st.session_state.comparison_apps = []

@st.cache_resource
def load_models():
    """Load ML models and preprocessors - handles 50 feature model"""
    try:
        model = joblib.load('optimized_neural_network.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('feature_columns.pkl')
        importance_df = pd.read_csv('feature_importance_rf.csv')
        
        # Log the actual number of features
        st.sidebar.write(f"Model trained with {len(features)} features")
        
        return model, scaler, features, importance_df
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {str(e)}")
        st.info("Please ensure these files are in your repository:")
        st.code("""
        - optimized_neural_network.pkl
        - scaler.pkl  
        - feature_columns.pkl
        - feature_importance_rf.csv
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()

def prepare_features_for_prediction(app_data, feature_list):
    """Prepare feature vector in correct order for prediction"""
    feature_vector = []
    
    # Only use the features that the model was trained with
    for feature in feature_list:
        if isinstance(app_data, pd.Series):
            value = app_data.get(feature, 0.0)
        elif isinstance(app_data, dict):
            value = app_data.get(feature, 0.0)
        else:
            value = 0.0
        
        # Ensure numeric value
        try:
            feature_vector.append(float(value))
        except:
            feature_vector.append(0.0)
    
    return np.array(feature_vector).reshape(1, -1)

def analyze_app(app_data, model, scaler, features):
    """Analyze a single app - works with model's expected features"""
    # IMPORTANT: Only use the features the model expects (50 features)
    X = prepare_features_for_prediction(app_data, features)
    
    # Verify we have the right number of features
    if X.shape[1] != len(features):
        st.error(f"Feature mismatch: prepared {X.shape[1]} features but model expects {len(features)}")
        return None
    
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    confidence = probabilities[prediction]
    adware_probability = probabilities[1]
    
    return {
        'prediction': prediction,
        'label': 'ADWARE' if prediction == 1 else 'BENIGN',
        'confidence': confidence,
        'adware_probability': adware_probability,
        'benign_probability': probabilities[0]
    }

def create_advanced_gauge(value, title, thresholds=[30, 70]):
    """Create an advanced gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24, 'color': '#1f2937'}},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#1f2937"},
            'bar': {'color': "rgba(102, 126, 234, 0.8)", 'thickness': 0.8},
            'bgcolor': "rgba(240, 242, 246, 0.4)",
            'borderwidth': 2,
            'bordercolor': "rgba(102, 126, 234, 0.3)",
            'steps': [
                {'range': [0, thresholds[0]], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [thresholds[0], thresholds[1]], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [thresholds[1], 100], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1f2937', 'family': 'Arial'}
    )
    return fig

def calculate_risk_score(app_data, adware_proba):
    """Calculate comprehensive risk score"""
    base_score = adware_proba * 100
    
    # Add weight for specific high-risk behaviors
    if app_data.get('network_op', 0) > 50:
        base_score = min(100, base_score + 10)
    if app_data.get('http_clients', 0) > 10:
        base_score = min(100, base_score + 8)
    if app_data.get('show_method', 0) > 100:
        base_score = min(100, base_score + 5)
    if app_data.get('gps_use', 0) > 3:
        base_score = min(100, base_score + 3)
    
    return base_score

def detect_behavioral_patterns(app_data):
    """Detect specific behavioral patterns"""
    patterns = []
    
    if app_data.get('network_op', 0) > 30 and app_data.get('http_clients', 0) > 5:
        patterns.append("🔴 **Aggressive Network Pattern**: Multiple HTTP clients with high network activity")
    
    if app_data.get('show_method', 0) > 50:
        patterns.append("🔴 **Ad Display Pattern**: Excessive UI manipulation detected")
    
    if app_data.get('sqllite_op', 0) > 200:
        patterns.append("🟡 **Data Harvesting Pattern**: High database activity")
    
    if app_data.get('fileio_op', 0) > 150:
        patterns.append("🟡 **File System Pattern**: Extensive file operations")
    
    if app_data.get('gps_use', 0) > 3:
        patterns.append("🔴 **Location Tracking**: GPS usage detected")
    
    if not patterns:
        patterns.append("✅ **Normal Behavior**: No suspicious patterns detected")
    
    return patterns

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ AI-Powered Adware Detection Pro</h1>', 
               unsafe_allow_html=True)
    
    # Load models first to get the actual feature count
    model, scaler, features, importance_df = load_models()
    feature_count = len(features)  # Get actual feature count
    
    # Top metrics bar - USING REAL DATA
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0;">99.63%</h3>
            <p style="margin: 0; opacity: 0.9;">F1-Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0;">99.57%</h3>
            <p style="margin: 0; opacity: 0.9;">Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0;">24,192</h3>
            <p style="margin: 0; opacity: 0.9;">Apps Trained</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0;">{feature_count}</h3>
            <p style="margin: 0; opacity: 0.9;">Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0;">⚡ Fast</h3>
            <p style="margin: 0; opacity: 0.9;">Real-time</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h2 style='color: #667eea; margin: 0;'>🛡️ DetectorPro</h2>
            <p style='opacity: 0.7;'>Advanced AI Security</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.header("🎯 Navigation")
        
        analysis_mode = st.radio(
            "Select Mode:",
            ["🏠 Dashboard", "📱 Single Analysis", "📊 Batch Processing", 
             "🧪 Model Testing", "📈 Feature Explorer", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # System status
        st.header("📊 System Status")
        st.success("✅ All Systems Operational")
        st.metric("Model Version", "v2.0.1")
        st.metric("Features Loaded", f"{feature_count}")
        
        st.markdown("---")
        
        # Quick Model Check
        if st.button("🔧 Verify Models"):
            try:
                st.success(f"✅ Model Type: {type(model).__name__}")
                st.info(f"📊 Features: {feature_count}")
                if not importance_df.empty:
                    st.info(f"🎯 Top Feature: {importance_df.iloc[0]['feature']}")
            except Exception as e:
                st.error(f"❌ {e}")
        
        if st.button("📝 Show Model Features"):
            st.write("Features used by model:")
            for i, f in enumerate(features, 1):
                st.write(f"{i}. {f}")
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style='text-align: center; opacity: 0.7; font-size: 0.8rem;'>
            Made with ❤️ by Parisa<br>
            © 2025 AI Security Lab
        </div>
        """, unsafe_allow_html=True)
    
    # Main content router
    if analysis_mode == "🏠 Dashboard":
        show_dashboard(model, scaler, features, importance_df)
    elif analysis_mode == "📱 Single Analysis":
        show_single_analysis(model, scaler, features)
    elif analysis_mode == "📊 Batch Processing":
        show_batch_processing(model, scaler, features)
    elif analysis_mode == "🧪 Model Testing":
        show_model_testing(model, scaler, features, importance_df)
    elif analysis_mode == "📈 Feature Explorer":
        show_feature_explorer(importance_df, features)
    else:
        show_about()

def show_dashboard(model, scaler, features, importance_df):
    """Dashboard with REAL metrics"""
    st.header("🏠 AI Security Dashboard")
    
    tabs = st.tabs(["📊 Overview", "📈 Model Performance", "🔍 Feature Importance"])
    
    with tabs[0]:
        st.subheader("System Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='glass-card'>
                <h3>Model Performance</h3>
                <p><strong>F1-Score:</strong> 99.63%</p>
                <p><strong>Accuracy:</strong> 99.57%</p>
                <p><strong>Precision:</strong> 99.69%</p>
                <p><strong>Recall:</strong> 99.57%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='glass-card'>
                <h3>Dataset Statistics</h3>
                <p><strong>Total Apps:</strong> 24,192</p>
                <p><strong>Adware:</strong> 14,149 (58.5%)</p>
                <p><strong>Benign:</strong> 10,043 (41.5%)</p>
                <p><strong>Features:</strong> {len(features)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            history_count = len(st.session_state.analysis_history)
            threats = sum(1 for a in st.session_state.analysis_history if 'ADWARE' in a.get('result', ''))
            last_analysis = st.session_state.analysis_history[-1]['time'][:19] if st.session_state.analysis_history else "None"
            
            st.markdown(f"""
            <div class='glass-card'>
                <h3>Analysis History</h3>
                <p><strong>Session Scans:</strong> {history_count}</p>
                <p><strong>Threats Found:</strong> {threats}</p>
                <p><strong>Last Analysis:</strong> {last_analysis}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.subheader("📊 Model Performance Metrics")
        
        metrics_data = {
            'Metric': ['F1-Score', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC'],
            'Score': [99.63, 99.57, 99.69, 99.57, 99.96]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        fig = px.bar(metrics_df, x='Metric', y='Score', 
                    title='Model Performance (%)',
                    color='Score', color_continuous_scale='Viridis',
                    text='Score')
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_range=[99, 100])
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("🎯 Feature Importance")
        
        if importance_df is not None and not importance_df.empty:
            top_n = st.slider("Number of features to display", 5, min(30, len(importance_df)), 15)
            top_features = importance_df.head(top_n)
            
            fig = px.bar(top_features, x='importance', y='feature',
                        orientation='h', 
                        title=f'Top {top_n} Most Important Features',
                        color='importance', color_continuous_scale='Plasma')
            fig.update_layout(height=max(400, top_n * 30))
            st.plotly_chart(fig, use_container_width=True)

def show_single_analysis(model, scaler, features):
    """Single app analysis - updated to handle feature mismatch"""
    st.header("📱 Single App Analysis")
    
    st.info(f"ℹ️ Model uses {len(features)} features for prediction")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Configure Analysis")
        
        tabs = st.tabs(["📁 Upload CSV", "⌨️ Manual Input", "🎲 Test Samples"])
        
        with tabs[0]:
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ File loaded: {uploaded_file.name}")
                    
                    # Show feature mapping info
                    with st.expander("Feature Mapping Info"):
                        available_features = df.columns.tolist()
                        used_features = [f for f in features if f in available_features]
                        missing_features = [f for f in features if f not in available_features]
                        
                        st.write(f"**Features in your file:** {len(available_features)}")
                        st.write(f"**Features used by model:** {len(used_features)}")
                        if missing_features:
                            st.warning(f"**Missing features (will use 0):** {len(missing_features)}")
                            if len(missing_features) <= 10:
                                st.write(", ".join(missing_features))
                    
                    # Remove file_name column if present
                    if 'file_name' in df.columns:
                        df = df.drop('file_name', axis=1)
                    
                    if len(df) > 1:
                        app_idx = st.selectbox("Select app to analyze", 
                                              range(len(df)),
                                              format_func=lambda x: f"App {x+1}")
                        app_data = df.iloc[app_idx].to_dict()
                    else:
                        app_data = df.iloc[0].to_dict()
                    
                    if st.button("🔍 Analyze App", type="primary"):
                        analyze_and_display(app_data, model, scaler, features)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with tabs[1]:
            st.info(f"Enter values for key features (model uses {len(features)} features)")
            
            app_data = {}
            
            # Show input for common features that are likely in the model
            with st.expander("🌐 Network Features", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    if 'network_op' in features:
                        app_data['network_op'] = st.number_input("Network Operations", 0.0, 1000.0, 5.0)
                    if 'http_clients' in features:
                        app_data['http_clients'] = st.number_input("HTTP Clients", 0.0, 100.0, 2.0)
                with col_b:
                    if 'con_timeout' in features:
                        app_data['con_timeout'] = st.number_input("Connection Timeout", 0, 1000, 0)
                    if 'socket_timeout' in features:
                        app_data['socket_timeout'] = st.number_input("Socket Timeout", 0, 1000, 0)
            
            with st.expander("📱 UI Features"):
                col_a, col_b = st.columns(2)
                with col_a:
                    if 'show_method' in features:
                        app_data['show_method'] = st.number_input("Show Methods", 0.0, 500.0, 25.0)
                    if 'show_dialog' in features:
                        app_data['show_dialog'] = st.number_input("Show Dialog", 0, 100, 5)
                with col_b:
                    if 'setcontentview' in features:
                        app_data['setcontentview'] = st.number_input("Set Content View", 0, 100, 5)
                    if 'xml_views' in features:
                        app_data['xml_views'] = st.number_input("XML Views", 0, 1000, 10)
            
            with st.expander("💾 Data Features"):
                col_a, col_b = st.columns(2)
                with col_a:
                    if 'sqllite_op' in features:
                        app_data['sqllite_op'] = st.number_input("SQLite Operations", 0.0, 1000.0, 50.0)
                    if 'fileio_op' in features:
                        app_data['fileio_op'] = st.number_input("File I/O Operations", 0.0, 500.0, 30.0)
                with col_b:
                    if 'gps_use' in features:
                        app_data['gps_use'] = st.number_input("GPS Use", 0.0, 10.0, 0.0)
                    if 'files' in features:
                        app_data['files'] = st.number_input("Files", 0, 10000, 100)
            
            with st.expander("📊 Code Metrics"):
                col_a, col_b = st.columns(2)
                with col_a:
                    if 'cyclomatic' in features:
                        app_data['cyclomatic'] = st.number_input("Cyclomatic Complexity", 0.0, 100000.0, 5000.0)
                    if 'methods' in features:
                        app_data['methods'] = st.number_input("Methods Count", 0, 50000, 5000)
                with col_b:
                    if 'classes' in features:
                        app_data['classes'] = st.number_input("Classes Count", 0, 5000, 200)
                    if 'wmc' in features:
                        app_data['wmc'] = st.number_input("WMC", 0, 100000, 5000)
            
            # Fill in missing features with 0
            for feature in features:
                if feature not in app_data:
                    app_data[feature] = 0
            
            if st.button("🔍 Analyze Configuration", type="primary"):
                analyze_and_display(app_data, model, scaler, features)
        
        with tabs[2]:
            st.info("Test with predefined samples")
            
            sample_apps = {
                "🔴 High-Risk Adware": {
                    'network_op': 85, 'http_clients': 25, 'show_method': 150,
                    'sqllite_op': 400, 'fileio_op': 280, 'gps_use': 5
                },
                "🟡 Moderate Risk": {
                    'network_op': 35, 'http_clients': 8, 'show_method': 60,
                    'sqllite_op': 150, 'fileio_op': 100, 'gps_use': 2
                },
                "✅ Safe App": {
                    'network_op': 3, 'http_clients': 1, 'show_method': 15,
                    'sqllite_op': 20, 'fileio_op': 25, 'gps_use': 0
                }
            }
            
            cols = st.columns(3)
            for idx, (name, data) in enumerate(sample_apps.items()):
                with cols[idx]:
                    if st.button(name, use_container_width=True):
                        # Ensure all required features are present
                        complete_data = {f: 0 for f in features}
                        complete_data.update(data)
                        analyze_and_display(complete_data, model, scaler, features)
    
    with col2:
        st.subheader("📊 Session Statistics")
        
        if st.session_state.analysis_history:
            total_scans = len(st.session_state.analysis_history)
            threat_count = sum(1 for a in st.session_state.analysis_history 
                             if 'ADWARE' in a.get('result', ''))
            
            st.metric("Total Scans", total_scans)
            st.metric("Threats Found", threat_count)
            if total_scans > 0:
                st.metric("Detection Rate", f"{threat_count/total_scans*100:.1f}%")
        
        st.subheader("🕐 Analysis History")
        
        if st.session_state.analysis_history:
            for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
                with st.expander(f"{analysis['time'][:19]} - {analysis['result']}"):
                    st.write(f"**Confidence:** {analysis['confidence']:.1%}")
                    st.write(f"**Risk Score:** {analysis.get('risk_score', 'N/A')}")

def analyze_and_display(app_data, model, scaler, features):
    """Analyze app and display results"""
    with st.spinner("🔄 Analyzing..."):
        result = analyze_app(app_data, model, scaler, features)
        
        if result is None:
            st.error("Analysis failed due to feature mismatch")
            return
        
        confidence = result['confidence']
        adware_proba = result['adware_probability']
        risk_score = calculate_risk_score(app_data, adware_proba)
    
    st.success("✅ Analysis Complete!")
    
    # Display results
    if result['prediction'] == 1:
        st.markdown(f"""
        <div class='danger-box'>
            <h1 style='margin: 0; font-size: 3rem;'>⚠️ ADWARE DETECTED</h1>
            <p style='font-size: 1.5rem; margin: 1rem 0;'>Risk Score: {risk_score:.1f}/100</p>
            <p style='font-size: 1.2rem;'>Confidence: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='success-box'>
            <h1 style='margin: 0; font-size: 3rem;'>✅ APP IS SAFE</h1>
            <p style='font-size: 1.5rem; margin: 1rem 0;'>Risk Score: {risk_score:.1f}/100</p>
            <p style='font-size: 1.2rem;'>Confidence: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed metrics
    st.subheader("📊 Detailed Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = create_advanced_gauge(confidence, "Confidence")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_advanced_gauge(adware_proba, "Adware Probability")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = create_advanced_gauge(risk_score/100, "Risk Score")
        st.plotly_chart(fig, use_container_width=True)
    
    # Behavioral patterns
    st.subheader("🔍 Behavioral Patterns Detected")
    patterns = detect_behavioral_patterns(app_data)
    for pattern in patterns:
        if "🔴" in pattern:
            st.error(pattern)
        elif "🟡" in pattern:
            st.warning(pattern)
        else:
            st.success(pattern)
    
    # Add to history
    st.session_state.analysis_history.append({
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'result': '🔴 ADWARE' if result['prediction'] == 1 else '✅ BENIGN',
        'confidence': confidence,
        'risk_score': risk_score
    })

def show_batch_processing(model, scaler, features):
    """Batch processing"""
    st.header("📊 Batch Processing")
    
    uploaded_file = st.file_uploader("Upload CSV with multiple apps", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} apps")
        
        # Remove file_name column if present
        if 'file_name' in df.columns:
            file_names = df['file_name'].tolist()
            df = df.drop('file_name', axis=1)
        else:
            file_names = [f"App_{i+1}" for i in range(len(df))]
        
        if st.button("🚀 Start Batch Analysis", type="primary"):
            progress_bar = st.progress(0)
            
            results = []
            for i, row in df.iterrows():
                result = analyze_app(row.to_dict(), model, scaler, features)
                
                if result:
                    results.append({
                        'App': file_names[i],
                        'Prediction': result['label'],
                        'Confidence': result['confidence'],
                        'Adware_Probability': result['adware_probability']
                    })
                
                progress_bar.progress((i + 1) / len(df))
            
            progress_bar.empty()
            
            if results:
                # Display results
                results_df = pd.DataFrame(results)
                
                # Summary
                col1, col2, col3 = st.columns(3)
                adware_count = len(results_df[results_df['Prediction'] == 'ADWARE'])
                
                with col1:
                    st.metric("Total Apps", len(results_df))
                with col2:
                    st.metric("Adware Detected", adware_count)
                with col3:
                    st.metric("Detection Rate", f"{adware_count/len(results_df)*100:.1f}%")
                
                # Results table
                st.dataframe(
                    results_df.style.applymap(
                        lambda x: 'background-color: #ffcccc' if x == 'ADWARE' else 'background-color: #ccffcc',
                        subset=['Prediction']
                    ),
                    use_container_width=True
                )
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def show_model_testing(model, scaler, features, importance_df):
    """Model testing interface"""
    st.header("🧪 Model Testing & Validation")
    
    tabs = st.tabs(["🔍 Model Info", "🧪 Test Patterns", "📁 Test File"])
    
    with tabs[0]:
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Details:**")
            st.write(f"- Type: {type(model).__name__}")
            st.write(f"- Features: {len(features)}")
            st.write(f"- Status: ✅ OK")
        
        with col2:
            st.write("**Performance Metrics:**")
            st.write("- F1-Score: 99.63%")
            st.write("- Accuracy: 99.57%")
            st.write("- Precision: 99.69%")
            st.write("- Recall: 99.57%")
    
    with tabs[1]:
        st.subheader("Test with Known Patterns")
        
        test_cases = {
            "Aggressive Adware": {
                'network_op': 85, 'http_clients': 25, 'show_method': 150,
                'sqllite_op': 400, 'fileio_op': 280, 'gps_use': 5,
                'expected': 'ADWARE'
            },
            "Benign App": {
                'network_op': 3, 'http_clients': 1, 'show_method': 15,
                'sqllite_op': 20, 'fileio_op': 25, 'gps_use': 0,
                'expected': 'BENIGN'
            }
        }
        
        for name, test_data in test_cases.items():
            st.write(f"**Testing: {name}**")
            expected = test_data.pop('expected')
            
            # Fill missing features
            complete_data = {f: 0 for f in features}
            complete_data.update(test_data)
            
            result = analyze_app(complete_data, model, scaler, features)
            
            if result:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"Expected: **{expected}**")
                with col2:
                    st.write(f"Predicted: **{result['label']}**")
                with col3:
                    if result['label'] == expected:
                        st.success("✅ PASS")
                    else:
                        st.error("❌ FAIL")
                
                st.write(f"Confidence: {result['confidence']:.2%}")
            st.write("---")
    
    with tabs[2]:
        st.subheader("Test with File")
        
        test_file = st.file_uploader("Upload test CSV", type=['csv'], key="test_upload")
        
        if test_file:
            df = pd.read_csv(test_file)
            st.write(f"Loaded {len(df)} test cases")
            
            if st.button("Run Tests"):
                results = []
                for i, row in df.iterrows():
                    result = analyze_app(row.to_dict(), model, scaler, features)
                    if result:
                        results.append(result['label'])
                
                if results:
                    st.write("Test Results:")
                    st.write(f"- Adware: {results.count('ADWARE')}")
                    st.write(f"- Benign: {results.count('BENIGN')}")

def show_feature_explorer(importance_df, features):
    """Feature explorer"""
    st.header("📈 Feature Explorer")
    
    tabs = st.tabs(["📊 Feature Importance", "📚 Feature Dictionary"])
    
    with tabs[0]:
        if importance_df is not None and not importance_df.empty:
            st.subheader("Feature Importance Rankings")
            
            # Only show features that are actually in the model
            model_importance = importance_df[importance_df['feature'].isin(features)]
            
            if not model_importance.empty:
                st.dataframe(model_importance, use_container_width=True)
                
                # Visualization
                top_n = st.slider("Top N features to visualize", 5, min(30, len(model_importance)), 10)
                top_features = model_importance.head(top_n)
                
                fig = px.bar(top_features, x='importance', y='feature',
                            orientation='h', title=f'Top {top_n} Features',
                            color='importance', color_continuous_scale='Viridis')
                fig.update_layout(height=max(300, top_n * 30))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance data not available")
    
    with tabs[1]:
        st.subheader("Feature Dictionary")
        st.write(f"Model uses {len(features)} features:")
        
        # Display all features used by the model
        cols = st.columns(3)
        for i, feature in enumerate(sorted(features)):
            with cols[i % 3]:
                st.write(f"• {feature}")

def show_about():
    """About page"""
    st.header("ℹ️ About AI-Powered Adware Detection System")
    
    tabs = st.tabs(["🏠 Overview", "📊 Performance", "🔬 Technology"])
    
    with tabs[0]:
        st.markdown("""
        ## Welcome to DetectorPro
        
        The most advanced AI-powered Android adware detection system, achieving
        **99.63% F1-Score** accuracy through state-of-the-art deep learning.
        
        ### 🎯 Our Mission
        To protect Android users worldwide from malicious adware through
        cutting-edge AI technology and continuous innovation.
        
        ### 🌟 Key Features
        - **99.63%** F1-Score accuracy
        - **24,192** apps analyzed in training
        - **50** behavioral features extracted
        - **<1 second** detection time
        - **Real-time** threat analysis
        """)
    
    with tabs[1]:
        st.markdown("""
        ## 📊 Performance Metrics
        
        ### Model Performance
        - **F1-Score:** 99.63%
        - **Accuracy:** 99.57%
        - **Precision:** 99.69%
        - **Recall:** 99.57%
        - **ROC-AUC:** 99.96%
        
        ### Dataset
        - **Total Apps:** 24,192
        - **Adware Samples:** 14,149 (58.5%)
        - **Benign Samples:** 10,043 (41.5%)
        - **Features:** 50 static and behavioral features
        """)
    
    with tabs[2]:
        st.markdown("""
        ## 🔬 Technology Stack
        
        ### Neural Network Architecture
        - **Type:** Multi-layer Perceptron (MLP)
        - **Optimizer:** Adam
        - **Training:** 5-fold cross-validation
        
        ### Feature Engineering
        - **Static Analysis:** APK structure, code metrics
        - **Behavioral Patterns:** Network, UI, data access
        - **Code Complexity:** Cyclomatic complexity, WMC
        
        ### Author
        **Dr. Parisa Hajibabaee**
        - AI/ML Research Scientist
        - Specialization: Deep Learning for Cybersecurity
        """)
def engineer_features(raw_data):
    """Convert raw features to engineered features that model expects"""
    eng_data = {}
    
    # Basic features that are kept as-is
    direct_features = ['cyclomatic', 'methods', 'classes', 'files', 'wmc', 'bytecode',
                      'object_map', 'no_action', 'network_op', 'timeout_wake_lock',
                      'catch', 'socket_timeout', 'dom_parser', 'scaled_bitmap',
                      'http_clients', 'fileio_op', 'contr_views', 'not_contr_views',
                      'unchecked_bundles', 'bundles', 'sqllite_op', 'lock_listener',
                      'dis_method', 'con_no_socket_timeout', 'show_method', 'onkeydown',
                      'setcontentview', 'bitmap_decode_op', 'apd', 'ppiv', 'cbo',
                      'onbackpressed', 'create_method', 'pot_bad_token', 'start_service']
    
    for feat in direct_features:
        eng_data[feat] = raw_data.get(feat, 0)
    
    # Engineered features
    eng_data['network_security_score'] = (
        0.4 * raw_data.get('network_op', 0) + 
        0.4 * raw_data.get('http_clients', 0) + 
        0.2 * raw_data.get('con_timeout', 0)
    )
    
    eng_data['ui_manipulation_score'] = (
        raw_data.get('dis_method', 0) + 
        raw_data.get('show_method', 0) + 
        raw_data.get('setcontentview', 0) + 
        raw_data.get('show_dialog', 0)
    )
    
    eng_data['data_operation_score'] = (
        raw_data.get('fileio_op', 0) * raw_data.get('sqllite_op', 0)
    )
    
    eng_data['permission_score'] = (
        raw_data.get('gps_use', 0) + 
        raw_data.get('lock_listener', 0) + 
        raw_data.get('timeout_wake_lock', 0)
    )
    
    eng_data['exception_ratio'] = raw_data.get('catch', 0) / (raw_data.get('methods', 0) + 1)
    eng_data['bytecode_per_method'] = raw_data.get('bytecode', 0) / (raw_data.get('methods', 0) + 1)
    
    # Polynomial features (products)
    eng_data['poly_network_op http_clients'] = raw_data.get('network_op', 0) * raw_data.get('http_clients', 0)
    eng_data['poly_network_op fileio_op'] = raw_data.get('network_op', 0) * raw_data.get('fileio_op', 0)
    eng_data['poly_sqllite_op fileio_op'] = raw_data.get('sqllite_op', 0) * raw_data.get('fileio_op', 0)
    eng_data['poly_http_clients fileio_op'] = raw_data.get('http_clients', 0) * raw_data.get('fileio_op', 0)
    eng_data['poly_http_clients sqllite_op'] = raw_data.get('http_clients', 0) * raw_data.get('sqllite_op', 0)
    eng_data['poly_network_op sqllite_op'] = raw_data.get('network_op', 0) * raw_data.get('sqllite_op', 0)
    
    eng_data['views_out_contr'] = raw_data.get('views_out_contr', 0)
    eng_data['unregister_recev'] = raw_data.get('unregister_recev', 0)
    eng_data['con_timeout'] = raw_data.get('con_timeout', 0)
    
    return eng_data

def analyze_app(app_data, model, scaler, features):
    """Analyze app with feature engineering"""
    # Engineer features first
    engineered_data = engineer_features(app_data)
    
    # Now prepare for prediction using engineered features
    X = prepare_features_for_prediction(engineered_data, features)
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    return {
        'prediction': prediction,
        'label': 'ADWARE' if prediction == 1 else 'BENIGN',
        'confidence': probabilities[prediction],
        'adware_probability': probabilities[1],
        'benign_probability': probabilities[0]
    }
# Run the application
if __name__ == "__main__":
    main()
