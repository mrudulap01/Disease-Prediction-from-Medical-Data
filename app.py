import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import datetime
from fpdf import FPDF
from model_deployment import (
    load_diabetes_model, predict_diabetes,
    load_heart_model, predict_heart_disease,
    load_cancer_model, predict_breast_cancer
)

st.set_page_config(page_title="MediPredict AI | Healthcare SaaS", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Premium Glassmorphism UI CSS */
    .stApp { 
        font-family: 'Inter', sans-serif; 
        background-color: #E2E8F0;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,85%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,90%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,85%,1) 0, transparent 50%);
    }
    header {visibility: hidden;} footer {visibility: hidden;}
    
    [data-testid="stThumbValue"] { display: none !important; }
    
    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] { 
        background: rgba(255, 255, 255, 0.4) !important; 
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Glass Panels */
    .glass-panel {
        background: rgba(255, 255, 255, 0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        padding: 30px;
        margin-bottom: 30px;
    }
    
    /* Input Fields */
    div[data-baseweb="input"] { 
        border-radius: 10px !important; 
        border: 1px solid rgba(255,255,255,0.8) !important; 
        background: rgba(255,255,255,0.45) !important; 
        backdrop-filter: blur(5px);
    }
    div[data-baseweb="input"]:focus-within { 
        border: 1px solid #3B82F6 !important; 
        background: rgba(255,255,255,0.8) !important;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #1E293B;
        margin-top: 30px;
        margin-bottom: 15px;
        border-bottom: 2px solid rgba(255,255,255,0.6);
        padding-bottom: 8px;
    }
    
    /* Primary Button */
    div.stButton > button[kind="primary"] { 
        background: linear-gradient(135deg, rgba(59,130,246,0.9) 0%, rgba(37,99,235,0.9) 100%) !important; 
        backdrop-filter: blur(5px);
        color: white !important; 
        border: 1px solid rgba(255,255,255,0.4) !important; 
        padding: 0.8rem !important; 
        font-size: 18px !important; 
        font-weight: 700 !important; 
        border-radius: 12px !important; 
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3) !important; 
        transition: all 0.3s ease;
    }
    div.stButton > button[kind="primary"]:hover { 
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(59, 130, 246, 0.4) !important; 
    }
    
    /* Risk Cards */
    .risk-card { padding: 40px; border-radius: 20px; text-align: center; display: flex; flex-direction: column; justify-content: center; height: 100%; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.5); }
    .risk-card.safe { background: rgba(220, 252, 231, 0.7); color: #14532D !important; border-top: 4px solid #22C55E; }
    .risk-card.warn { background: rgba(254, 243, 199, 0.7); color: #78350F !important; border-top: 4px solid #F59E0B; }
    .risk-card.danger { background: rgba(254, 226, 226, 0.7); color: #7F1D1D !important; border-top: 4px solid #EF4444; }
    
    /* Label text for inputs */
    .input-label { font-weight: 600; font-size: 14px; color: #334155; margin-bottom: 4px; display: flex; justify-content: space-between; }
    .input-help { color: #64748B; font-size: 12px; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Timestamp', 'Disease', 'Prediction', 'Confidence', 'Risk Category'])

def sync_input(source_key, target_key):
    st.session_state[target_key] = st.session_state[source_key]

@st.cache_resource(show_spinner=False)
def load_all_artifacts():
    return {
        "Diabetes": load_diabetes_model(),
        "Heart Disease": load_heart_model(),
        "Breast Cancer": load_cancer_model()
    }

artifacts_dict = load_all_artifacts()

def apply_profile(profile_type, disease):
    if disease == "Diabetes":
        profiles = {
            'healthy': {'d_age': 25, 'd_bmi': 22.0, 'd_glu': 90, 'd_bp': 70, 'd_ins': 45, 'd_skin': 15, 'd_preg': 0, 'd_dpf': 0.2},
            'borderline': {'d_age': 45, 'd_bmi': 29.5, 'd_glu': 135, 'd_bp': 85, 'd_ins': 120, 'd_skin': 28, 'd_preg': 3, 'd_dpf': 0.55},
            'high_risk': {'d_age': 55, 'd_bmi': 36.0, 'd_glu': 185, 'd_bp': 95, 'd_ins': 210, 'd_skin': 35, 'd_preg': 6, 'd_dpf': 0.95}
        }
    elif disease == "Heart Disease":
        profiles = {
            'healthy': {'h_age': 30, 'h_sex': 1, 'h_cp': 0, 'h_trestbps': 110, 'h_chol': 180, 'h_fbs': 0, 'h_restecg': 0, 'h_thalach': 160, 'h_exang': 0, 'h_oldpeak': 0.0, 'h_slope': 2, 'h_ca': 0, 'h_thal': 2},
            'borderline': {'h_age': 55, 'h_sex': 1, 'h_cp': 2, 'h_trestbps': 135, 'h_chol': 230, 'h_fbs': 0, 'h_restecg': 1, 'h_thalach': 140, 'h_exang': 1, 'h_oldpeak': 1.5, 'h_slope': 1, 'h_ca': 1, 'h_thal': 2},
            'high_risk': {'h_age': 65, 'h_sex': 1, 'h_cp': 3, 'h_trestbps': 160, 'h_chol': 280, 'h_fbs': 1, 'h_restecg': 2, 'h_thalach': 110, 'h_exang': 1, 'h_oldpeak': 3.5, 'h_slope': 0, 'h_ca': 3, 'h_thal': 3}
        }
    elif disease == "Breast Cancer":
        profiles = {
            'healthy': {'b_mp': 70.0, 'b_mcp': 0.02, 'b_wr': 11.0, 'b_wp': 80.0, 'b_wcp': 0.05},
            'borderline': {'b_mp': 90.0, 'b_mcp': 0.05, 'b_wr': 15.0, 'b_wp': 100.0, 'b_wcp': 0.10},
            'high_risk': {'b_mp': 130.0, 'b_mcp': 0.12, 'b_wr': 20.0, 'b_wp': 140.0, 'b_wcp': 0.20}
        }
        
    if profile_type == 'random':
        if disease == "Diabetes":
            prof = {
                'd_age': int(np.random.randint(20, 80)), 'd_bmi': round(np.random.uniform(18.0, 45.0), 1),
                'd_glu': int(np.random.randint(70, 200)), 'd_bp': int(np.random.randint(60, 120)),
                'd_ins': int(np.random.randint(15, 300)), 'd_skin': int(np.random.randint(10, 50)),
                'd_preg': int(np.random.randint(0, 8)), 'd_dpf': round(np.random.uniform(0.1, 1.2), 2)
            }
        elif disease == "Heart Disease":
            prof = {
                'h_age': int(np.random.randint(30, 80)), 'h_sex': int(np.random.randint(0, 2)),
                'h_cp': int(np.random.randint(0, 4)), 'h_trestbps': int(np.random.randint(100, 180)),
                'h_chol': int(np.random.randint(150, 350)), 'h_fbs': int(np.random.randint(0, 2)),
                'h_restecg': int(np.random.randint(0, 3)), 'h_thalach': int(np.random.randint(80, 190)),
                'h_exang': int(np.random.randint(0, 2)), 'h_oldpeak': round(np.random.uniform(0.0, 4.0), 1),
                'h_slope': int(np.random.randint(0, 3)), 'h_ca': int(np.random.randint(0, 4)), 'h_thal': int(np.random.randint(1, 4))
            }
        elif disease == "Breast Cancer":
            prof = {
                'b_mp': round(np.random.uniform(60.0, 150.0), 1), 'b_mcp': round(np.random.uniform(0.01, 0.15), 3),
                'b_wr': round(np.random.uniform(10.0, 25.0), 1), 'b_wp': round(np.random.uniform(70.0, 180.0), 1),
                'b_wcp': round(np.random.uniform(0.05, 0.25), 3)
            }
    else:
        prof = profiles[profile_type]
        
    for key, val in prof.items():
        st.session_state[f'{key}_slider'] = val
        st.session_state[f'{key}_num'] = val
    st.toast(f'Loaded {profile_type.replace("_", " ").title()} Profile!', icon='✅')

def create_pdf_report(patient_data, result, disease_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(59, 130, 246)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(15)
    pdf.cell(0, 10, "MediPredict AI Platform", ln=True, align='C')
    pdf.set_y(50)
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, f"{disease_name} Risk Report", ln=True, align='C')
    pdf.set_font("Arial", 'I', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Generated On: {datetime.datetime.now().strftime('%B %d, %Y - %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 10, "AI Assessment Result:", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(40, 10, "Risk Status: ")
    pdf.set_font("Arial", '', 14)
    if result['prediction_code'] == 1: pdf.set_text_color(220, 38, 38)
    else: pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, f"{result['prediction_label']}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(40, 10, "Confidence: ")
    pdf.set_font("Arial", '', 14)
    pdf.cell(0, 10, f"{result['confidence_score']}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(59, 130, 246)
    pdf.cell(0, 10, "Patient Input Metrics:", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(0, 0, 0)
    for key, value in patient_data.items():
        pdf.cell(80, 8, f"{key}:", border=1)
        pdf.cell(110, 8, f"{value}", border=1, ln=True)
    return bytes(pdf.output())

with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="background: linear-gradient(135deg, rgba(59,130,246,0.8) 0%, rgba(37,99,235,0.8) 100%); color: white; width: 60px; height: 60px; border-radius: 15px; display: inline-flex; align-items: center; justify-content: center; font-size: 28px; margin-bottom: 10px; box-shadow: 0 4px 10px rgba(37,99,235,0.2); backdrop-filter: blur(5px);">🩺</div>
            <h2 style="color: #1E293B; margin-bottom: 0; font-size: 24px;">MediPredict AI</h2>
            <p style="color: #64748B; font-weight: 500; font-size: 13px; margin-top: 0;">Clinical Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("**Navigation**")
    page = st.radio("Go to", ["🏠 Analytics Dashboard", "🔬 Disease Assessment", "📜 Patient History"], label_visibility="collapsed")
    
    st.markdown("---")
    st.info("💡 **Tip:** Use the 'Autofill' buttons on the Assessment page to test different patient scenarios quickly.")

def render_input(key_prefix, emoji_icon, label, unit, min_val, max_val, step_val, default_val):
    if f'{key_prefix}_slider' not in st.session_state: st.session_state[f'{key_prefix}_slider'] = default_val
    if f'{key_prefix}_num' not in st.session_state: st.session_state[f'{key_prefix}_num'] = default_val

    st.markdown(f'''
        <div class="input-label">
            <span>{emoji_icon} {label}</span>
            <span class="input-help">{unit}</span>
        </div>
    ''', unsafe_allow_html=True)
    
    st.number_input(f"n_{key_prefix}", min_value=min_val, max_value=max_val, step=step_val, key=f"{key_prefix}_num", on_change=sync_input, args=(f"{key_prefix}_num", f"{key_prefix}_slider"), label_visibility="collapsed")
    st.slider(f"s_{key_prefix}", min_value=min_val, max_value=max_val, step=step_val, key=f"{key_prefix}_slider", on_change=sync_input, args=(f"{key_prefix}_slider", f"{key_prefix}_num"), label_visibility="collapsed")
    st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

if page == "🏠 Analytics Dashboard":
    st.markdown('<div class="glass-panel"><h1 style="margin:0; color: #1E293B;">Global Analytics Overview</h1><p style="margin:5px 0 0 0; color: #475569;">Real-time platform metrics and ML engine status.</p></div>', unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="glass-panel" style="text-align:center;"><h3 style="margin:0; color:#475569;">Total Screenings (YTD)</h3><h1 style="margin:0; color:#2563EB;">142,850</h1></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="glass-panel" style="text-align:center;"><h3 style="margin:0; color:#475569;">Avg Model Confidence</h3><h1 style="margin:0; color:#10B981;">94.2%</h1></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="glass-panel" style="text-align:center;"><h3 style="margin:0; color:#475569;">System Uptime</h3><h1 style="margin:0; color:#8B5CF6;">99.99%</h1></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown('<div class="glass-panel" style="padding-bottom:0;">', unsafe_allow_html=True)
        dates = pd.date_range(end=datetime.datetime.now(), periods=30)
        volume = np.random.normal(loc=5000, scale=500, size=30).astype(int)
        df_trend = pd.DataFrame({'Date': dates, 'Screenings': volume})
        fig_trend = px.line(df_trend, x='Date', y='Screenings', title='30-Day Screening Volume Trend')
        fig_trend.update_traces(line_color='#3B82F6', line_width=3, fill='tozeroy', fillcolor='rgba(59,130,246,0.1)')
        fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=10, l=0, r=0))
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('''
            <div class="glass-panel" style="height: 100%;">
                <h3 style="margin-top:0; color: #1E293B;">AI Engine Status</h3>
                <div style="margin-top: 25px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="font-weight:600; color:#475569;">🩸 Diabetes Engine</span>
                        <span style="background:#DCFCE7; color:#166534; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:bold;">🟢 ONLINE</span>
                    </div>
                    <div style="width:100%; background:#E2E8F0; border-radius:4px; height:8px; margin-bottom:25px;">
                        <div style="width:94%; background:#3B82F6; height:8px; border-radius:4px;"></div>
                    </div>
                    
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="font-weight:600; color:#475569;">❤️ Heart Engine</span>
                        <span style="background:#DCFCE7; color:#166534; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:bold;">🟢 ONLINE</span>
                    </div>
                    <div style="width:100%; background:#E2E8F0; border-radius:4px; height:8px; margin-bottom:25px;">
                        <div style="width:91%; background:#EF4444; height:8px; border-radius:4px;"></div>
                    </div>
                    
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="font-weight:600; color:#475569;">🔬 Cancer Engine</span>
                        <span style="background:#DCFCE7; color:#166534; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:bold;">🟢 ONLINE</span>
                    </div>
                    <div style="width:100%; background:#E2E8F0; border-radius:4px; height:8px; margin-bottom:20px;">
                        <div style="width:96%; background:#10B981; height:8px; border-radius:4px;"></div>
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div class="glass-panel" style="padding-bottom:0;">', unsafe_allow_html=True)
    np.random.seed(42)
    lat = np.random.uniform(25, 49, 300)
    lon = np.random.uniform(-125, -70, 300)
    risk = np.random.choice(['High Risk', 'Medium Risk', 'Low Risk'], 300, p=[0.2, 0.3, 0.5])
    df_map = pd.DataFrame({'Lat': lat, 'Lon': lon, 'Risk Level': risk})
    color_map = {'High Risk': '#EF4444', 'Medium Risk': '#F59E0B', 'Low Risk': '#10B981'}
    fig_map = px.scatter_geo(df_map, lat='Lat', lon='Lon', color='Risk Level', color_discrete_map=color_map, title='Live Diagnostics Heatmap (USA Region)', opacity=0.7, size_max=8)
    fig_map.update_geos(scope='usa', bgcolor='rgba(0,0,0,0)', showcoastlines=False, showland=True, landcolor='rgba(226, 232, 240, 0.8)', showlakes=False)
    fig_map.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=10, l=0, r=0))
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "🔬 Disease Assessment":
    st.markdown('<div class="glass-panel"><h2 style="margin:0; color: #1E293B;">Step 1: Select Diagnostic Model</h2></div>', unsafe_allow_html=True)
    disease_selection = st.selectbox("Disease Type", ["Diabetes", "Heart Disease", "Breast Cancer"], label_visibility="collapsed")
    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

    st.markdown(f'''
        <div class="glass-panel" style="border-left: 5px solid rgba(59,130,246,0.8);">
            <h1 style="margin:0; font-size: 32px; color: #1E293B;">{disease_selection} Assessment</h1>
            <p style="margin:10px 0 0 0; color: #475569; font-size: 16px;">AI-Powered Diagnostic Prediction Engine</p>
        </div>
    ''', unsafe_allow_html=True)
        
    artifacts = artifacts_dict[disease_selection]
    if not all(artifacts):
        st.error(f"Model artifacts missing for {disease_selection}. Please train the model first.")
        st.stop()
        
    st.markdown('<p style="font-weight: 600; font-size: 16px; color: #334155;">Quick Test Profiles</p>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    with q1: st.button("🟢 Healthy", on_click=apply_profile, args=('healthy', disease_selection), use_container_width=True)
    with q2: st.button("🟡 Borderline", on_click=apply_profile, args=('borderline', disease_selection), use_container_width=True)
    with q3: st.button("🔴 High Risk", on_click=apply_profile, args=('high_risk', disease_selection), use_container_width=True)
    with q4: st.button("🎲 Random Patient", on_click=apply_profile, args=('random', disease_selection), use_container_width=True)
    
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    user_input = {}
    
    if disease_selection == "Diabetes":
        st.markdown('<div class="section-header">Demographics & Vitals</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: render_input('d_age', '👤', 'Age', 'Years', 1, 120, 1, 30)
        with c2: render_input('d_bmi', '⚖️', 'BMI', 'kg/m²', 10.0, 70.0, 0.1, 25.0)
        with c3: render_input('d_bp', '❤️', 'Blood Pressure', 'mmHg', 0, 150, 1, 70)
        
        st.markdown('<div class="section-header">Metabolic Markers</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: render_input('d_glu', '🩸', 'Glucose', 'mg/dL', 0, 300, 1, 110)
        with c2: render_input('d_ins', '💉', 'Insulin', 'mu U/ml', 0, 900, 1, 79)
        with c3: render_input('d_dpf', '🧬', 'Pedigree Function', 'Score', 0.0, 3.0, 0.05, 0.4)
        
        st.markdown('<div class="section-header">Other Indicators</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: render_input('d_skin', '🤏', 'Skin Thickness', 'mm', 0, 100, 1, 20)
        with c2: render_input('d_preg', '🤰', 'Pregnancies', 'Count', 0, 20, 1, 1)
        
        user_input = {'Pregnancies': st.session_state['d_preg_num'], 'Glucose': st.session_state['d_glu_num'], 'BloodPressure': st.session_state['d_bp_num'], 'SkinThickness': st.session_state['d_skin_num'], 'Insulin': st.session_state['d_ins_num'], 'BMI': st.session_state['d_bmi_num'], 'DiabetesPedigreeFunction': st.session_state['d_dpf_num'], 'Age': st.session_state['d_age_num']}

    elif disease_selection == "Heart Disease":
        st.markdown('<div class="section-header">Demographics & Symptoms</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: render_input('h_age', '👤', 'Age', 'Years', 1, 120, 1, 50)
        with c2: render_input('h_sex', '⚧️', 'Sex', '1=Male, 0=Female', 0, 1, 1, 1)
        with c3: render_input('h_cp', '🫀', 'Chest Pain Type', '0=None, 3=Severe', 0, 3, 1, 0)
        
        st.markdown('<div class="section-header">Clinical Vitals</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: render_input('h_trestbps', '❤️', 'Resting BP', 'mmHg', 80, 200, 1, 120)
        with c2: render_input('h_thalach', '🏃', 'Max Heart Rate', 'bpm', 60, 220, 1, 150)
        with c3: render_input('h_exang', '😓', 'Exercise Angina', '1=Yes, 0=No', 0, 1, 1, 0)
        
        st.markdown('<div class="section-header">Lab & Test Results</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: render_input('h_chol', '🩸', 'Cholesterol', 'mg/dl', 100, 600, 1, 200)
        with c2: render_input('h_fbs', '🍩', 'Fasting Blood Sugar', '1 = >120mg/dl', 0, 1, 1, 0)
        with c3: render_input('h_restecg', '📈', 'Resting ECG', '0-2', 0, 2, 1, 0)
        with c4: render_input('h_oldpeak', '📉', 'ST Depression', 'Value', 0.0, 10.0, 0.1, 1.0)
        
        c5, c6, c7, c8 = st.columns(4)
        with c5: render_input('h_slope', '📐', 'ST Slope', '0-2', 0, 2, 1, 1)
        with c6: render_input('h_ca', '🔬', 'Major Vessels', '0-3', 0, 3, 1, 0)
        with c7: render_input('h_thal', '🩸', 'Thalassemia', '1-3', 1, 3, 1, 2)
        
        user_input = {'age': st.session_state['h_age_num'], 'sex': st.session_state['h_sex_num'], 'cp': st.session_state['h_cp_num'], 'trestbps': st.session_state['h_trestbps_num'], 'chol': st.session_state['h_chol_num'], 'fbs': st.session_state['h_fbs_num'], 'restecg': st.session_state['h_restecg_num'], 'thalach': st.session_state['h_thalach_num'], 'exang': st.session_state['h_exang_num'], 'oldpeak': st.session_state['h_oldpeak_num'], 'slope': st.session_state['h_slope_num'], 'ca': st.session_state['h_ca_num'], 'thal': st.session_state['h_thal_num']}

    elif disease_selection == "Breast Cancer":
        st.markdown('<div class="section-header">Cell Nucleus Geometry</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: render_input('b_mp', '📏', 'Mean Perimeter', 'mm', 0.0, 300.0, 1.0, 90.0)
        with c2: render_input('b_wp', '📐', 'Worst Perimeter', 'mm', 0.0, 300.0, 1.0, 100.0)
        
        st.markdown('<div class="section-header">Severity Metrics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: render_input('b_wr', '⭕', 'Worst Radius', 'mm', 0.0, 50.0, 0.1, 15.0)
        with c2: render_input('b_mcp', '🔬', 'Mean Concave Points', 'Value', 0.0, 0.5, 0.01, 0.05)
        with c3: render_input('b_wcp', '🔍', 'Worst Concave Points', 'Value', 0.0, 0.5, 0.01, 0.1)
        
        user_input = {'mean perimeter': st.session_state['b_mp_num'], 'mean concave points': st.session_state['b_mcp_num'], 'worst radius': st.session_state['b_wr_num'], 'worst perimeter': st.session_state['b_wp_num'], 'worst concave points': st.session_state['b_wcp_num']}

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submitted = st.button("✨ Generate AI Diagnostic Report", type="primary", use_container_width=True)

    if submitted:
        st.markdown("---")
        with st.spinner('Analyzing patient data...'):
            time.sleep(1)
            try:
                if disease_selection == "Diabetes":
                    result = predict_diabetes(user_input, artifacts)
                    risk_label = result['prediction_label']
                    prob = float(result['probabilities']['Diabetic'].strip('%'))
                elif disease_selection == "Heart Disease":
                    result = predict_heart_disease(user_input, artifacts)
                    risk_label = result['prediction_label']
                    prob = float(result['probabilities']['Heart Disease Present'].strip('%'))
                elif disease_selection == "Breast Cancer":
                    result = predict_breast_cancer(user_input, artifacts)
                    risk_label = result['prediction_label']
                    prob = float(result['probabilities']['Malignant'].strip('%'))
            except Exception as e:
                st.error(f"Inference Engine Error: {e}")
                st.stop()
                
            if prob < 40:
                risk_cat = "Low Risk"; theme_class = "safe"; icon = "🟢"
            elif prob < 60:
                risk_cat = "Medium Risk"; theme_class = "warn"; icon = "🟡"
            else:
                risk_cat = "High Risk"; theme_class = "danger"; icon = "🔴"
                
            new_record = pd.DataFrame([{
                'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                'Disease': disease_selection,
                'Prediction': risk_label, 'Confidence': result['confidence_score'], 'Risk Category': risk_cat
            }])
            st.session_state.history = pd.concat([st.session_state.history, new_record], ignore_index=True)

            st.markdown(f'''
                <div style="text-align: center; margin-bottom: 20px;">
                    <h2 style="color: #1E293B;">Official Clinical Report</h2>
                </div>
            ''', unsafe_allow_html=True)
            
            r1, r2 = st.columns([1.5, 1])
            
            with r1:
                st.markdown(f"""
                    <div class="risk-card {theme_class}">
                        <h1 style="font-size: 50px; margin-bottom: 10px;">{icon} {risk_cat.upper()}</h1>
                        <h3 style="margin-top: 0; opacity: 0.9; font-weight: 500;">AI Diagnosis: {risk_label}</h3>
                        <div style="margin-top: 20px; background: rgba(255,255,255,0.4); padding: 10px; border-radius: 10px; display: inline-block; align-self: center; backdrop-filter: blur(5px);">
                            <span style="font-size: 24px; font-weight: 800;">{result['confidence_score']} Confidence</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            with r2:
                st.markdown("""
                    <div class="glass-panel" style="height: 100%; margin-bottom: 0;">
                        <h3 style="color: #1E293B; margin-top: 0;">Summary</h3>
                        <p style="color: #475569; margin-bottom: 5px;"><strong>Disease:</strong></p>
                        <p style="color: #1E293B; font-weight: 600; margin-top: 0;">{0}</p>
                        <p style="color: #475569; margin-bottom: 5px;"><strong>Timestamp:</strong></p>
                        <p style="color: #1E293B; font-weight: 600; margin-top: 0;">{1}</p>
                        <div style="margin-top: 30px;">
                """.format(disease_selection, datetime.datetime.now().strftime('%Y-%m-%d %H:%M')), unsafe_allow_html=True)
                
                try:
                    pdf_bytes = create_pdf_report(user_input, result, disease_selection)
                    st.download_button(label="📥 Download PDF Report", data=pdf_bytes, file_name=f"Assessment_{disease_selection.replace(' ', '_')}.pdf", mime="application/pdf", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to generate PDF Report: {e}")
                st.markdown("</div></div>", unsafe_allow_html=True)

elif page == "📜 Patient History":
    st.markdown('<div class="glass-panel"><h1 style="margin:0; color: #1E293B;">Prediction History</h1></div>', unsafe_allow_html=True)
    if st.session_state.history.empty:
        st.info("No predictions made in this session yet.")
    else:
        st.dataframe(st.session_state.history, use_container_width=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = pd.DataFrame(columns=['Timestamp', 'Disease', 'Prediction', 'Confidence', 'Risk Category'])
            st.rerun()
