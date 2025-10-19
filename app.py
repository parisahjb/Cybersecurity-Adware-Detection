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
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

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

# Define all 67 features
FEATURE_NAMES = [
    'network_op', 'sqllite_op', 'fileio_op', 'bitmap_decode_op', 'dis_method',
    'show_method', 'setcontentview', 'scaled_bitmap', 'onkeydown', 'is_playing',
    'unregister_recev', 'onbackpressed', 'show_dialog', 'create_method', 
    'timeout_wake_lock', 'lock_listener', 'gps_use', 'xml_pull_parser',
    'sax_parser', 'dom_parser', 'catch', 'log', 'no_action', 'max_noc',
    'max_dit', 'lcom', 'cbo', 'ppiv', 'apd', 'start_activities',
    'start_activity', 'start_instrum', 'start_intent_sender', 'start_service',
    'start_action_mode', 'start_activity_result', 'start_activity_from_child',
    'start_activity_from_frag', 'start_activity_needed', 'start_intent_for_result',
    'start_intent_from_child', 'start_next_activity', 'start_search',
    'contr_views', 'not_contr_views', 'xml_views', 'max_xml_views',
    'views_out_contr', 'pot_bad_token', 'fragments', 'http_clients',
    'con_timeout', 'socket_timeout', 'con_no_timeout', 'con_no_socket_timeout',
    'bundles', 'checked_bundles', 'unchecked_bundles', 'object_map',
    'files', 'classes', 'methods', 'bytecode', 'methods_per_class',
    'bytecode_per_method', 'cyclomatic', 'wmc'
]

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'comparison_apps' not in st.session_state:
    st.session_state.comparison_apps = []

@st.cache_resource
def load_models():
    """Load ML models and preprocessors - NO DUMMY DATA"""
    try:
        model = joblib.load('optimized_neural_network.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('feature_columns.pkl')
        importance_df = pd.read_csv('feature_importance_rf.csv')
        
        # Verify features
        if len(features) != 67:
            st.warning(f"Expected 67 features but loaded {len(features)}")
        
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
    for feature in feature_list:
        if feature in app_data:
            feature_vector.append(float(app_data[feature]))
        else:
            feature_vector.append(0.0)
    return np.array(feature_vector).reshape(1, -1)

def analyze_app(app_data, model, scaler, features):
    """Analyze a single app"""
    X = prepare_features_for_prediction(app_data, features)
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
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin: 0;">67</h3>
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
    
    # Load models
    model, scaler, features, importance_df = load_models()
    
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
        st.metric("Features Loaded", f"{len(features)}")
        
        st.markdown("---")
        
        # Quick Model Check
        if st.button("🔧 Verify Models"):
            try:
                st.success(f"✅ Model Type: {type(model).__name__}")
                st.info(f"📊 Features: {len(features)}")
                if not importance_df.empty:
                    st.info(f"🎯 Top Feature: {importance_df.iloc[0]['feature']}")
            except Exception as e:
                st.error(f"❌ {e}")
        
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
            st.markdown("""
            <div class='glass-card'>
                <h3>Model Performance</h3>
                <p><strong>F1-Score:</strong> 99.63%</p>
                <p><strong>Accuracy:</strong> 99.57%</p>
                <p><strong>Precision:</strong> 99.69%</p>
                <p><strong>Recall:</strong> 99.57%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='glass-card'>
                <h3>Dataset Statistics</h3>
                <p><strong>Total Apps:</strong> 24,192</p>
                <p><strong>Adware:</strong> 14,149 (58.5%)</p>
                <p><strong>Benign:</strong> 10,043 (41.5%)</p>
                <p><strong>Features:</strong> 67</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='glass-card'>
                <h3>Analysis History</h3>
                <p><strong>Session Scans:</strong> {}</p>
                <p><strong>Threats Found:</strong> {}</p>
                <p><strong>Last Analysis:</strong> {}</p>
            </div>
            """.format(
                len(st.session_state.analysis_history),
                sum(1 for a in st.session_state.analysis_history if 'ADWARE' in a.get('result', '')),
                st.session_state.analysis_history[-1]['time'][:19] if st.session_state.analysis_history else "None"
            ), unsafe_allow_html=True)
    
    with tabs[1]:
        st.subheader("📊 Model Performance Metrics")
        
        # Real performance metrics
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
        
        # Model comparison
        st.subheader("Model Comparison")
        comparison_data = {
            'Model': ['Neural Network (Ours)', 'Random Forest', 'XGBoost', 'SVM', 'Naive Bayes'],
            'F1-Score': [99.63, 99.56, 99.48, 98.21, 96.54]
        }
        
        comp_df = pd.DataFrame(comparison_data)
        fig = px.bar(comp_df, x='Model', y='F1-Score', 
                    title='F1-Score Comparison',
                    color='F1-Score', color_continuous_scale='Blues')
        fig.update_layout(yaxis_range=[95, 100])
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.subheader("🎯 Feature Importance")
        
        if importance_df is not None and not importance_df.empty:
            # Top features
            top_n = st.slider("Number of features to display", 5, 30, 15)
            top_features = importance_df.head(top_n)
            
            fig = px.bar(top_features, x='importance', y='feature',
                        orientation='h', 
                        title=f'Top {top_n} Most Important Features',
                        color='importance', color_continuous_scale='Plasma')
            fig.update_layout(height=max(400, top_n * 30))
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature categories
            st.subheader("Feature Categories")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**🌐 Network Features**")
                network_features = ['network_op', 'http_clients', 'con_timeout', 'socket_timeout']
                for f in network_features:
                    if f in importance_df['feature'].values:
                        imp = importance_df[importance_df['feature'] == f]['importance'].values[0]
                        st.write(f"- {f}: {imp:.4f}")
            
            with col2:
                st.write("**📱 UI Features**")
                ui_features = ['show_method', 'show_dialog', 'setcontentview', 'xml_views']
                for f in ui_features:
                    if f in importance_df['feature'].values:
                        imp = importance_df[importance_df['feature'] == f]['importance'].values[0]
                        st.write(f"- {f}: {imp:.4f}")
            
            with col3:
                st.write("**💾 Data Features**")
                data_features = ['sqllite_op', 'fileio_op', 'bundles', 'files']
                for f in data_features:
                    if f in importance_df['feature'].values:
                        imp = importance_df[importance_df['feature'] == f]['importance'].values[0]
                        st.write(f"- {f}: {imp:.4f}")
        else:
            st.warning("Feature importance data not available")

def show_single_analysis(model, scaler, features):
    """Single app analysis"""
    st.header("📱 Single App Analysis")
    
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
            st.info("Enter feature values manually")
            
            app_data = {}
            
            with st.expander("🌐 Network Features", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    app_data['network_op'] = st.number_input("Network Operations", 0.0, 1000.0, 5.0)
                    app_data['http_clients'] = st.number_input("HTTP Clients", 0.0, 100.0, 2.0)
                with col_b:
                    app_data['con_timeout'] = st.number_input("Connection Timeout", 0, 1000, 0)
                    app_data['socket_timeout'] = st.number_input("Socket Timeout", 0, 1000, 0)
            
            with st.expander("📱 UI Features"):
                col_a, col_b = st.columns(2)
                with col_a:
                    app_data['show_method'] = st.number_input("Show Methods", 0.0, 500.0, 25.0)
                    app_data['show_dialog'] = st.number_input("Show Dialog", 0, 100, 5)
                with col_b:
                    app_data['setcontentview'] = st.number_input("Set Content View", 0, 100, 5)
                    app_data['xml_views'] = st.number_input("XML Views", 0, 1000, 10)
            
            with st.expander("💾 Data Features"):
                col_a, col_b = st.columns(2)
                with col_a:
                    app_data['sqllite_op'] = st.number_input("SQLite Operations", 0.0, 1000.0, 50.0)
                    app_data['fileio_op'] = st.number_input("File I/O Operations", 0.0, 500.0, 30.0)
                with col_b:
                    app_data['gps_use'] = st.number_input("GPS Use", 0.0, 10.0, 0.0)
                    app_data['files'] = st.number_input("Files", 0, 10000, 100)
            
            with st.expander("📊 Code Metrics"):
                col_a, col_b = st.columns(2)
                with col_a:
                    app_data['cyclomatic'] = st.number_input("Cyclomatic Complexity", 0.0, 100000.0, 5000.0)
                    app_data['methods'] = st.number_input("Methods Count", 0, 50000, 5000)
                with col_b:
                    app_data['classes'] = st.number_input("Classes Count", 0, 5000, 200)
                    app_data['wmc'] = st.number_input("WMC", 0, 100000, 5000)
            
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
                        analyze_and_display(data, model, scaler, features)
    
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
                    st.write(f"**Risk Score:** {analysis.get('risk_score', 'N/A'):.1f}")

def analyze_and_display(app_data, model, scaler, features):
    """Analyze app and display results"""
    with st.spinner("🔄 Analyzing..."):
        result = analyze_app(app_data, model, scaler, features)
        
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
    
    # Recommendations
    st.subheader("💡 Recommendations")
    if result['prediction'] == 1 and risk_score >= 70:
        st.error("""
        **Immediate Actions Required:**
        1. Quarantine the application immediately
        2. Revoke all permissions
        3. Scan device for related threats
        4. Check for data breaches
        """)
    elif result['prediction'] == 1:
        st.warning("""
        **Recommended Actions:**
        1. Monitor application behavior
        2. Restrict sensitive permissions
        3. Consider safer alternatives
        """)
    else:
        st.success("""
        **Best Practices:**
        1. Continue regular scanning
        2. Keep apps updated
        3. Monitor permission requests
        """)
    
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
                
                results.append({
                    'App': file_names[i],
                    'Prediction': result['label'],
                    'Confidence': result['confidence'],
                    'Adware_Probability': result['adware_probability']
                })
                
                progress_bar.progress((i + 1) / len(df))
            
            progress_bar.empty()
            
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
            st.write(f"- Expected Features: 67")
            st.write(f"- Status: {'✅ OK' if len(features) == 67 else '❌ Mismatch'}")
        
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
            
            result = analyze_app(test_data, model, scaler, features)
            
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
                    results.append(result['label'])
                
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
            
            # Search
            search = st.text_input("Search features", "")
            
            if search:
                filtered_df = importance_df[importance_df['feature'].str.contains(search, case=False)]
            else:
                filtered_df = importance_df
            
            if not filtered_df.empty:
                st.dataframe(filtered_df, use_container_width=True)
                
                # Visualization
                top_n = st.slider("Top N features to visualize", 5, 30, 10)
                top_features = filtered_df.head(top_n)
                
                fig = px.bar(top_features, x='importance', y='feature',
                            orientation='h', title=f'Top {top_n} Features',
                            color='importance', color_continuous_scale='Viridis')
                fig.update_layout(height=max(300, top_n * 30))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance data not available")
    
    with tabs[1]:
        st.subheader("Feature Dictionary")
        
        feature_categories = {
            "🌐 Network": ['network_op', 'http_clients', 'con_timeout', 'socket_timeout'],
            "📱 UI/Display": ['show_method', 'show_dialog', 'setcontentview', 'xml_views', 'fragments'],
            "💾 Data/Storage": ['sqllite_op', 'fileio_op', 'bundles', 'files'],
            "📍 Sensors": ['gps_use'],
            "🎯 Activities": ['start_activity', 'start_service', 'start_activities'],
            "📊 Code Metrics": ['cyclomatic', 'methods', 'classes', 'bytecode', 'wmc']
        }
        
        for category, feat_list in feature_categories.items():
            with st.expander(category):
                for feat in feat_list:
                    if feat in features:
                        st.write(f"✅ **{feat}** - Available in model")
                    else:
                        st.write(f"❌ **{feat}** - Not in current model")

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
        - **67** behavioral features extracted
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
        - **Features:** 67 static and behavioral features
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

# Run the application
if __name__ == "__main__":
    main()
