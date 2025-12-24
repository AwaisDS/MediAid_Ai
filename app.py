import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hashlib
import os
import sqlite3
import datetime
import json
import base64
import pickle

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & STYLING
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="MediAid AI",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Modern React-like CSS
modern_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    color: #E8EAED;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Enhanced Buttons */
.stButton>button {
    background: linear-gradient(135deg, #00D9A3 0%, #00B4D8 100%);
    color: #000000;
    border-radius: 12px;
    font-weight: 600;
    font-size: 15px;
    border: none;
    padding: 14px 28px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 20px rgba(0, 217, 163, 0.25);
    letter-spacing: 0.5px;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #00B88F 0%, #0096C7 100%);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 217, 163, 0.4);
    color: #000000;
}

.stButton>button:active {
    transform: translateY(0px);
}

/* Cards */
.big-card {
    background: rgba(26, 31, 58, 0.6);
    backdrop-filter: blur(10px);
    padding: 32px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    margin-bottom: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    transition: all 0.3s ease;
}

.big-card:hover {
    border-color: rgba(0, 217, 163, 0.3);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
}

.metric-card {
    background: linear-gradient(135deg, rgba(0, 217, 163, 0.08) 0%, rgba(0, 180, 216, 0.08) 100%);
    border-left: 4px solid #00D9A3;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 16px;
    transition: all 0.3s ease;
}

.metric-card:hover {
    background: linear-gradient(135deg, rgba(0, 217, 163, 0.12) 0%, rgba(0, 180, 216, 0.12) 100%);
    transform: translateX(4px);
}

/* Typography */
.title {
    font-size: 56px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #00D9A3 0%, #00B4D8 50%, #E8EAED 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 12px;
    letter-spacing: -1px;
    line-height: 1.2;
}

.subtitle {
    text-align: center;
    color: #B0B8C1;
    font-size: 20px;
    margin-bottom: 40px;
    font-weight: 400;
    letter-spacing: 0.3px;
}

.section-header {
    font-size: 32px;
    font-weight: 700;
    color: #E8EAED;
    margin-top: 24px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid rgba(0, 217, 163, 0.3);
    letter-spacing: -0.5px;
}

/* Input Fields */
.stTextInput>div>div>input, 
.stTextArea>div>div>textarea,
.stNumberInput>div>div>input,
.stSelectbox>div>div>select {
    background: rgba(26, 31, 58, 0.8) !important;
    border: 2px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #E8EAED !important;
    padding: 12px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus,
.stNumberInput>div>div>input:focus,
.stSelectbox>div>div>select:focus {
    border-color: #00D9A3 !important;
    box-shadow: 0 0 0 3px rgba(0, 217, 163, 0.1) !important;
}

/* Labels */
.stTextInput>label, 
.stTextArea>label,
.stNumberInput>label,
.stSelectbox>label,
.stSlider>label,
.stCheckbox>label {
    color: #E8EAED !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    margin-bottom: 8px !important;
    letter-spacing: 0.3px !important;
}

/* Checkboxes */
.stCheckbox>label>div {
    background: rgba(26, 31, 58, 0.8);
    border: 2px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    padding: 8px 12px;
    transition: all 0.3s ease;
}

.stCheckbox>label>div:hover {
    border-color: #00D9A3;
    background: rgba(0, 217, 163, 0.05);
}

/* Sliders */
.stSlider>div>div>div>div {
    background: #00D9A3 !important;
}

/* Info boxes */
.stAlert {
    border-radius: 12px;
    border: none;
    padding: 16px 20px;
    font-weight: 500;
    backdrop-filter: blur(10px);
}

.stSuccess {
    background: rgba(16, 185, 129, 0.15) !important;
    border-left: 4px solid #10B981 !important;
}

.stInfo {
    background: rgba(59, 130, 246, 0.15) !important;
    border-left: 4px solid #3B82F6 !important;
}

.stWarning {
    background: rgba(245, 158, 11, 0.15) !important;
    border-left: 4px solid #F59E0B !important;
}

.stError {
    background: rgba(239, 68, 68, 0.15) !important;
    border-left: 4px solid #EF4444 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(26, 31, 58, 0.6);
    border-radius: 10px;
    padding: 12px 24px;
    color: #B0B8C1;
    font-weight: 600;
    border: 2px solid transparent;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(0, 217, 163, 0.1);
    color: #00D9A3;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 217, 163, 0.2) 0%, rgba(0, 180, 216, 0.2) 100%) !important;
    border-color: #00D9A3 !important;
    color: #00D9A3 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(26, 31, 58, 0.6);
    border-radius: 12px;
    border: 2px solid rgba(255, 255, 255, 0.08);
    color: #E8EAED !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 16px !important;
    transition: all 0.3s ease;
}

.streamlit-expanderHeader:hover {
    border-color: #00D9A3;
    background: rgba(0, 217, 163, 0.05);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1f3a 0%, #0f1419 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

section[data-testid="stSidebar"] .stButton>button {
    width: 100%;
    text-align: left;
    background: rgba(26, 31, 58, 0.6);
    color: #E8EAED;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 8px;
}

section[data-testid="stSidebar"] .stButton>button:hover {
    background: rgba(0, 217, 163, 0.15);
    border-color: #00D9A3;
    color: #00D9A3;
}

/* Dataframe */
.dataframe {
    background: rgba(26, 31, 58, 0.6) !important;
    color: #E8EAED !important;
    border-radius: 12px !important;
}

/* Footer */
.footer {
    color: #6B7280;
    text-align: center;
    font-size: 13px;
    margin-top: 60px;
    padding-top: 24px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    font-weight: 400;
}

/* Custom Action Cards */
.action-card {
    background: rgba(26, 31, 58, 0.6);
    backdrop-filter: blur(10px);
    padding: 24px;
    border-radius: 16px;
    border: 2px solid rgba(255, 255, 255, 0.08);
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
}

.action-card:hover {
    border-color: #00D9A3;
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 217, 163, 0.2);
}

/* Form Container */
.stForm {
    background: rgba(26, 31, 58, 0.4);
    padding: 24px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* Download Link */
a {
    color: #00D9A3 !important;
    text-decoration: none !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

a:hover {
    color: #00B88F !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(26, 31, 58, 0.4);
}

::-webkit-scrollbar-thumb {
    background: rgba(0, 217, 163, 0.3);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 217, 163, 0.5);
}
</style>
"""
st.markdown(modern_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATABASE MANAGEMENT (SQLite)
# -----------------------------------------------------------------------------

DB_FILE = "mediaid_db.db"

def init_db():
    """Initialize database with all required tables"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Users Table
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        password TEXT,
                        email TEXT,
                        is_admin INTEGER DEFAULT 0,
                        join_date TEXT
                    )''')
        
        # Medical Profile Table
        c.execute('''CREATE TABLE IF NOT EXISTS profiles (
                        username TEXT PRIMARY KEY,
                        age INTEGER,
                        gender TEXT,
                        height REAL,
                        weight REAL,
                        blood_group TEXT,
                        allergies TEXT,
                        chronic_diseases TEXT,
                        medications TEXT,
                        surgeries TEXT
                    )''')
        
        # Reports/History Table
        c.execute('''CREATE TABLE IF NOT EXISTS reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT,
                        date TEXT,
                        symptoms TEXT,
                        prediction TEXT,
                        confidence TEXT,
                        recommendations TEXT
                    )''')
                    
        # Educational Articles
        c.execute('''CREATE TABLE IF NOT EXISTS articles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT,
                        content TEXT,
                        category TEXT
                    )''')
        
        # Seed Admin if not exists
        c.execute("SELECT * FROM users WHERE username = 'admin'")
        if not c.fetchone():
            admin_pass = hashlib.sha256(str.encode("admin123")).hexdigest()
            c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", 
                      ('admin', admin_pass, 'admin@mediaid.com', 1, str(datetime.date.today())))
            
            # Seed sample articles
            articles = [
                ("Preventing Seasonal Flu", "Wash hands frequently with soap for at least 20 seconds. Get vaccinated annually. Avoid close contact with sick people. Cover your mouth when coughing or sneezing.", "Prevention"),
                ("Recognizing Dehydration", "Common symptoms include dry mouth, fatigue, dark urine, dizziness, and reduced urination. Drink water immediately and increase fluid intake. Seek medical attention if severe.", "General Health"),
                ("First Aid for Cuts", "Clean the wound thoroughly with water. Apply antibiotic ointment to prevent infection. Cover with a clean bandage. Change dressing daily and watch for signs of infection.", "First Aid"),
                ("Managing Fever", "Rest adequately and stay hydrated. Take fever-reducing medication as directed. Monitor temperature regularly. Seek medical care if fever exceeds 103°F (39.4°C) or persists.", "General Health"),
                ("Healthy Sleep Habits", "Maintain a consistent sleep schedule. Create a relaxing bedtime routine. Keep your bedroom cool and dark. Avoid screens 1 hour before bed. Aim for 7-9 hours of sleep.", "Prevention")
            ]
            c.executemany("INSERT INTO articles (title, content, category) VALUES (?, ?, ?)", articles)

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database initialization error: {e}")
        return False

# Initialize DB on app start
init_db()

# DB Helper Functions
def run_query(query, params=(), fetch_one=False, fetch_all=False, commit=False):
    """Execute database queries safely"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(query, params)
        data = None
        if fetch_one:
            data = c.fetchone()
        elif fetch_all:
            data = c.fetchall()
        if commit:
            conn.commit()
        conn.close()
        return data
    except Exception as e:
        st.error(f"Database error: {e}")
        return None

def make_hashes(password):
    """Hash password using SHA256"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_user(username, password):
    """Verify user credentials"""
    hashed = make_hashes(password)
    user = run_query("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed), fetch_one=True)
    return user

def create_user(username, password, email):
    """Create new user account"""
    if run_query("SELECT * FROM users WHERE username = ?", (username,), fetch_one=True):
        return False
    hashed = make_hashes(password)
    run_query("INSERT INTO users (username, password, email, join_date) VALUES (?, ?, ?, ?)", 
              (username, hashed, email, str(datetime.date.today())), commit=True)
    return True

# -----------------------------------------------------------------------------
# 3. MODEL LOADING & LOGIC
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__) if os.path.dirname(__file__) else "."
MODEL_PATH = os.path.join(BASE_DIR, "Prototype_model", "medi_aid_disease_model.pkl")
LE_PATH = os.path.join(BASE_DIR, "Prototype_model", "medi_aid_label_encoders.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Prototype_model", "medi_aid_scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "Prototype_model", "feature_names.pkl")

# Load Resources
model, label_encoders, scaler, feature_names = None, None, None, None

def load_models():
    """Load ML models with error handling"""
    global model, label_encoders, scaler, feature_names
    try:
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
            except:
                with open(MODEL_PATH, "rb") as f:
                    model = pickle.load(f)

        if os.path.exists(LE_PATH):
            try:
                label_encoders = joblib.load(LE_PATH)
            except:
                with open(LE_PATH, "rb") as f:
                    label_encoders = pickle.load(f)

        if os.path.exists(SCALER_PATH):
            try:
                scaler = joblib.load(SCALER_PATH)
            except:
                with open(SCALER_PATH, "rb") as f:
                    scaler = pickle.load(f)

        if os.path.exists(FEATURE_NAMES_PATH):
            with open(FEATURE_NAMES_PATH, "rb") as f:
                feature_names = pickle.load(f)
                
        return True
    except Exception as e:
        st.error(f"⚠️ Error loading AI models: {e}")
        return False

load_models()

# Full expected feature list
EXPECTED_FEATURES = [
    'age','gender','region','duration_days','comorbidity',
    'abdominal_pain','back_pain','chest_pain','chills','conjunctivitis','constipation','cough','dark_urine',
    'dehydration','diarrhea','dysuria','fatigue','fever','frequency','headache','itching','joint_pain',
    'jaundice','loss_of_appetite','loss_of_smell_taste','lower_abdominal_pain','muscle_pain','nausea',
    'night_sweats','persistent_cough','rash','retro_orbital_pain','runny_nose','shortness_of_breath',
    'sore_throat','sputum','sweating','vesicular_rash','vomiting','weight_loss'
]

# -----------------------------------------------------------------------------
# 4. SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "Landing"

def navigate_to(page):
    """Navigate to a different page"""
    st.session_state.current_page = page
    st.rerun()

def logout():
    """Logout user and clear session"""
    st.session_state.logged_in = False
    st.session_state.user_info = None
    st.session_state.current_page = "Landing"
    if "last_result" in st.session_state:
        del st.session_state.last_result
    st.rerun()

# -----------------------------------------------------------------------------
# 5. PAGE FUNCTIONS
# -----------------------------------------------------------------------------

# --- PUBLIC PAGES ---

def render_landing():
    """Landing page with login/signup options"""
    st.markdown("<div class='title'>💉 MediAid AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Your Intelligent Health Companion Powered by Advanced Machine Learning</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 40px; color: #B0B8C1; font-size: 17px; line-height: 1.8;'>
            <p style='margin-bottom: 20px;'>MediAid AI uses cutting-edge machine learning algorithms to analyze your symptoms 
            and provide preliminary health insights with confidence scores.</p>
            <p style='color: #00D9A3; font-weight: 600;'>✓ Instant symptom analysis &nbsp;&nbsp; ✓ Personalized health reports &nbsp;&nbsp; ✓ Medical history tracking</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔐 Login to Your Account", use_container_width=True):
                navigate_to("Login")
        with col_b:
            if st.button("📝 Create New Account", use_container_width=True):
                navigate_to("Signup")

    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("⚠️ **Important Medical Disclaimer:** MediAid AI is a preliminary diagnostic tool and does NOT replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns. In case of emergency, contact your local emergency services immediately (e.g., 911, 1122).")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Feature highlights
    feat1, feat2, feat3, feat4 = st.columns(4)
    with feat1:
        st.info("**🤖 AI-Powered**\n\nAdvanced ML algorithms")
    with feat2:
        st.success("**📊 Detailed Reports**\n\nComprehensive analysis")
    with feat3:
        st.warning("**🔒 Secure Data**\n\nPrivacy protected")
    with feat4:
        st.error("**📚 Health Library**\n\nEducational resources")

def render_login():
    """User login page"""
    st.markdown("<div class='section-header'>🔐 Login to Your Account</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='big-card'>", unsafe_allow_html=True)
        
        username = st.text_input("Username", placeholder="Enter your username", key="login_user")
        password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("✅ Sign In", use_container_width=True, type="primary"):
            if not username or not password:
                st.error("❌ Please fill in all fields")
            else:
                user = verify_user(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_info = user
                    st.success("✅ Login successful! Redirecting...")
                    navigate_to("Dashboard")
                else:
                    st.error("❌ Invalid username or password. Please try again.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_x, col_y = st.columns(2)
        with col_x:
            if st.button("📝 Create Account", use_container_width=True):
                navigate_to("Signup")
        with col_y:
            if st.button("← Back to Home", use_container_width=True):
                navigate_to("Landing")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='text-align: center; margin-top: 20px; color: #6B7280; font-size: 13px;'>Demo: username: admin | password: admin123</div>", unsafe_allow_html=True)

def render_signup():
    """User registration page"""
    st.markdown("<div class='section-header'>📝 Create Your Account</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='big-card'>", unsafe_allow_html=True)
        
        new_user = st.text_input("Username", placeholder="Choose a unique username", key="signup_user")
        new_email = st.text_input("Email Address", placeholder="your.email@example.com", key="signup_email")
        new_pass = st.text_input("Password", type="password", placeholder="Create a strong password", key="signup_pass")
        confirm_pass = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password", key="signup_confirm")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        agree = st.checkbox("I agree to the Medical Disclaimer and Terms of Service", key="agree_terms")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Create Account", use_container_width=True, type="primary", disabled=not agree):
            if not new_user or not new_email or not new_pass or not confirm_pass:
                st.error("❌ All fields are required")
            elif new_pass != confirm_pass:
                st.error("❌ Passwords do not match")
            elif len(new_pass) < 6:
                st.error("❌ Password must be at least 6 characters long")
            elif "@" not in new_email:
                st.error("❌ Please enter a valid email address")
            else:
                if create_user(new_user, new_pass, new_email):
                    st.success("✅ Account created successfully! Please login.")
                    st.balloons()
                    navigate_to("Login")
                else:
                    st.error("❌ Username already exists. Please choose a different username.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("← Already have an account? Login", use_container_width=True):
            navigate_to("Login")
        
        st.markdown("</div>", unsafe_allow_html=True)

# --- USER PRIVATE PAGES ---

def render_dashboard():
    """Main user dashboard"""
    user = st.session_state.user_info[0]
    st.markdown(f"<div class='title'>Welcome back, {user} 👋</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Your personal health companion dashboard</div>", unsafe_allow_html=True)
    
    # 4 Main Action Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='action-card'>", unsafe_allow_html=True)
        st.markdown("### 🏥")
        st.markdown("**Start Diagnosis**")
        st.markdown("<p style='color: #B0B8C1; font-size: 13px;'>Check your symptoms now</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Check Symptoms", use_container_width=True, key="dash_symptoms"):
            navigate_to("SymptomCheck")
    
    with col2:
        st.markdown("<div class='action-card'>", unsafe_allow_html=True)
        st.markdown("### 👤")
        st.markdown("**Medical Profile**")
        st.markdown("<p style='color: #B0B8C1; font-size: 13px;'>Manage your health info</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Update Profile", use_container_width=True, key="dash_profile"):
            navigate_to("Profile")
    
    with col3:
        st.markdown("<div class='action-card'>", unsafe_allow_html=True)
        st.markdown("### 📜")
        st.markdown("**Reports History**")
        st.markdown("<p style='color: #B0B8C1; font-size: 13px;'>View past diagnoses</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("View Reports", use_container_width=True, key="dash_history"):
            navigate_to("History")
    
    with col4:
        st.markdown("<div class='action-card'>", unsafe_allow_html=True)
        st.markdown("### 📚")
        st.markdown("**Health Education**")
        st.markdown("<p style='color: #B0B8C1; font-size: 13px;'>Learn about health</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Health Articles", use_container_width=True, key="dash_education"):
            navigate_to("Education")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Statistics Overview
    st.markdown("<div class='section-header'>📊 Your Health Overview</div>", unsafe_allow_html=True)
    
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    # Get user statistics
    report_count = run_query("SELECT COUNT(*) FROM reports WHERE username = ?", (user,), fetch_one=True)
    total_reports = report_count[0] if report_count else 0
    
    profile_data = run_query("SELECT * FROM profiles WHERE username = ?", (user,), fetch_one=True)
    profile_complete = "Complete" if profile_data else "Incomplete"
    
    with stats_col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"### {total_reports}")
        st.markdown("**Total Diagnoses**")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with stats_col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"### {profile_complete}")
        st.markdown("**Profile Status**")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with stats_col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### Active")
        st.markdown("**Account Status**")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Daily Health Tip
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='big-card'>", unsafe_allow_html=True)
    st.markdown("### 💡 Daily Health Tip")
    st.markdown("""
    **Stay hydrated throughout the day!** Drinking adequate water helps regulate body temperature, 
    keeps joints lubricated, prevents infections, delivers nutrients to cells, and keeps organs functioning properly. 
    Aim for 8-10 glasses of water daily.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

def render_profile():
    """Medical profile management page"""
    st.markdown("<div class='section-header'>👤 Medical Profile</div>", unsafe_allow_html=True)
    username = st.session_state.user_info[0]
    
    # Load existing profile data
    data = run_query("SELECT * FROM profiles WHERE username = ?", (username,), fetch_one=True)
    
    # Set default values
    age, gender, height, weight, bg = 25, "Male", 170.0, 70.0, "O+"
    allergies, chronic, meds, surgs = "None", "None", "None", "None"
    
    if data:
        age, gender, height, weight, bg, allergies, chronic, meds, surgs = data[1:]

    st.info("📝 Complete your medical profile to get more personalized health recommendations")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.form("profile_form"):
        st.markdown("### Basic Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age (years)", 1, 120, int(age))
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                                index=["Male", "Female", "Other"].index(gender) if gender in ["Male", "Female", "Other"] else 0)
        with col3:
            blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], 
                                      index=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"].index(bg) if bg in ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"] else 6)
        
        col4, col5 = st.columns(2)
        with col4:
            height = st.number_input("Height (cm)", 50.0, 250.0, float(height), step=0.1)
        with col5:
            weight = st.number_input("Weight (kg)", 20.0, 200.0, float(weight), step=0.1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Medical History")
        
        col6, col7 = st.columns(2)
        with col6:
            allergies = st.text_area("Known Allergies", value=allergies, height=100,
                                    placeholder="e.g., Penicillin, Peanuts, Dust")
            chronic_diseases = st.text_area("Chronic Conditions", value=chronic, height=100,
                                           placeholder="e.g., Diabetes, Hypertension, Asthma")
        with col7:
            medications = st.text_area("Current Medications", value=meds, height=100,
                                      placeholder="e.g., Metformin 500mg, Aspirin")
            surgeries = st.text_area("Past Surgeries", value=surgs, height=100,
                                    placeholder="e.g., Appendectomy (2015)")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.form_submit_button("💾 Save Profile", use_container_width=True, type="primary"):
            try:
                run_query("REPLACE INTO profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                         (username, age, gender, height, weight, blood_group, 
                          allergies, chronic_diseases, medications, surgeries), commit=True)
                st.success("✅ Profile updated successfully!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Error saving profile: {e}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Dashboard"):
        navigate_to("Dashboard")

def render_symptom_check():
    """Symptom checker interface"""
    st.markdown("<div class='section-header'>🩺 AI Symptom Checker</div>", unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ AI model files are missing. Please ensure model files are in the 'Prototype_model' directory.")
        if st.button("← Back to Dashboard"):
            navigate_to("Dashboard")
        return

    st.info("ℹ️ Answer the questions below to receive an AI-powered preliminary health assessment")
    
    username = st.session_state.user_info[0]
    prof = run_query("SELECT * FROM profiles WHERE username = ?", (username,), fetch_one=True)
    
    default_age = prof[1] if prof else 25
    default_gender = prof[2] if prof else "Male"
    
    # Step 1: Patient Details
    with st.expander("📋 Step 1: Patient Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 1, 100, int(default_age))
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                                index=["Male", "Female", "Other"].index(default_gender))
        with col3:
            region = st.selectbox("Region/Province", 
                                ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan", 
                                 "Islamabad", "Gilgit-Baltistan", "Azad Kashmir"])
        
        col4, col5 = st.columns(2)
        with col4:
            duration_days = st.number_input("Symptom Duration (Days)", 1, 30, 3)
        with col5:
            comorbidity = st.selectbox("Pre-existing Conditions", 
                                      ["None", "Diabetes", "Hypertension", "Chronic Lung Disease", 
                                       "HIV", "Heart Disease"])

    # Step 2: Symptom Selection
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 Step 2: Select Your Symptoms")
    st.caption("Check all symptoms you are currently experiencing")
    
    symptom_list = EXPECTED_FEATURES[5:]
    
    # Categorize symptoms for better UX
    categories = {
        "General Symptoms": ['fever', 'fatigue', 'chills', 'sweating', 'dehydration', 'weight_loss', 'loss_of_appetite'],
        "Pain & Discomfort": ['headache', 'joint_pain', 'muscle_pain', 'back_pain', 'chest_pain', 
                             'abdominal_pain', 'lower_abdominal_pain', 'retro_orbital_pain'],
        "Respiratory & ENT": ['cough', 'persistent_cough', 'sore_throat', 'runny_nose', 
                             'shortness_of_breath', 'sputum', 'loss_of_smell_taste'],
        "Digestive System": ['nausea', 'vomiting', 'diarrhea', 'constipation', 'jaundice'],
        "Skin & Other": ['rash', 'itching', 'vesicular_rash', 'dark_urine', 'dysuria', 
                        'frequency', 'conjunctivitis', 'night_sweats']
    }
    
    selected_symptoms = []
    
    tabs = st.tabs(list(categories.keys()))
    
    for i, (cat_name, cat_syms) in enumerate(categories.items()):
        with tabs[i]:
            cols = st.columns(3)
            for j, sym in enumerate(cat_syms):
                if sym in symptom_list:
                    symptom_label = sym.replace("_", " ").title()
                    is_present = cols[j % 3].checkbox(symptom_label, key=f"chk_{sym}")
                    if is_present:
                        selected_symptoms.append(sym)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
    with col_btn2:
        analyze_btn = st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True)
    
    if analyze_btn:
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom to proceed with the analysis")
        else:
            with st.spinner("🤖 AI is analyzing your symptoms... Please wait"):
                process_prediction(age, gender, region, duration_days, comorbidity, selected_symptoms)

def process_prediction(age, gender, region, duration_days, comorbidity, selected_symptoms):
    """Process ML prediction from user inputs"""
    try:
        # Prepare input data
        row = {s: (1 if s in selected_symptoms else 0) for s in EXPECTED_FEATURES 
               if s not in ['age', 'gender', 'region', 'duration_days', 'comorbidity']}
        row.update({
            'age': age, 
            'gender': gender, 
            'region': region, 
            'duration_days': duration_days, 
            'comorbidity': comorbidity
        })
        
        df_input = pd.DataFrame([row])
        
        # Ensure all features exist
        for col in EXPECTED_FEATURES:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[EXPECTED_FEATURES]
        
        # Encode categorical variables
        for col in ['gender', 'region', 'comorbidity']:
            if label_encoders and col in label_encoders:
                try:
                    df_input[col] = label_encoders[col].transform(df_input[col])
                except:
                    df_input[col] = 0
            else:
                df_input[col] = 0
        
        # Scale numerical features
        if scaler:
            try:
                df_input[['age', 'duration_days']] = scaler.transform(df_input[['age', 'duration_days']])
            except:
                pass
        
        # Make prediction
        prediction_cls = model.predict(df_input.values.reshape(1, -1))[0]
        
        # Get probability scores
        top_3 = []
        try:
            probs = model.predict_proba(df_input.values.reshape(1, -1))[0]
            classes = model.classes_
            sorted_indices = np.argsort(probs)[::-1]
            for i in range(min(3, len(classes))):
                idx = sorted_indices[i]
                top_3.append((classes[idx], probs[idx] * 100))
        except:
            top_3.append((prediction_cls, 100.0))
        
        # Save results
        result_data = {
            "top_3": top_3,
            "symptoms": selected_symptoms,
            "date": str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "advice": get_medical_advice(top_3[0][0])
        }
        
        # Save to database
        run_query("""INSERT INTO reports (username, date, symptoms, prediction, confidence, recommendations) 
                     VALUES (?, ?, ?, ?, ?, ?)""", 
                 (st.session_state.user_info[0], result_data['date'], ",".join(selected_symptoms), 
                  top_3[0][0], f"{top_3[0][1]:.1f}%", json.dumps(result_data['advice'])), 
                 commit=True)
        
        st.session_state.last_result = result_data
        navigate_to("Results")
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.info("Please try again or contact support if the problem persists")

def get_medical_advice(disease):
    """Get medical advice for specific disease"""
    advice_database = {
        "Dengue": {
            "tests": "NS1 Antigen Test, Complete Blood Count (CBC) with Platelet Count",
            "meds": "Paracetamol for fever, Adequate hydration, Rest",
            "emergency": "Severe abdominal pain, persistent vomiting, bleeding gums, blood in stool/vomit"
        },
        "Malaria": {
            "tests": "Blood Smear Test, Rapid Diagnostic Test (RDT), Microscopy",
            "meds": "Antimalarial medication (prescribed by doctor), Fever management",
            "emergency": "Confusion, seizures, difficulty breathing, severe anemia, jaundice"
        },
        "Typhoid": {
            "tests": "Widal Test, Blood Culture, Stool Culture",
            "meds": "Antibiotics (prescribed), Hydration, Nutritious diet",
            "emergency": "Severe abdominal pain, intestinal bleeding, confusion, high-grade persistent fever"
        },
        "COVID-19": {
            "tests": "RT-PCR Test, Rapid Antigen Test, Chest X-Ray if severe",
            "meds": "Isolation, Supportive care, Plenty of fluids, Paracetamol if needed",
            "emergency": "Difficulty breathing, persistent chest pain, confusion, bluish lips"
        },
        "Pneumonia": {
            "tests": "Chest X-Ray, Blood Tests, Sputum Culture",
            "meds": "Antibiotics (prescribed), Rest, Hydration",
            "emergency": "Severe difficulty breathing, blue lips/fingernails, confusion, chest pain"
        },
        "Influenza": {
            "tests": "Clinical diagnosis, Rapid Influenza Test if needed",
            "meds": "Rest, Fluids, Paracetamol for fever, Antivirals if prescribed",
            "emergency": "Difficulty breathing, chest pain, severe weakness, confusion"
        },
        "Tuberculosis": {
            "tests": "Chest X-Ray, Sputum Test, Tuberculin Skin Test",
            "meds": "Anti-TB medication (6-9 months course as prescribed)",
            "emergency": "Coughing blood, severe chest pain, extreme weight loss"
        }
    }
    
    return advice_database.get(disease, {
        "tests": "Consult a healthcare provider for appropriate diagnostic tests",
        "meds": "Consult a doctor for proper treatment plan",
        "emergency": "Severe worsening of symptoms, difficulty breathing, loss of consciousness"
    })

def render_results():
    """Display prediction results and recommendations"""
    if "last_result" not in st.session_state:
        navigate_to("Dashboard")
        return
    
    res = st.session_state.last_result
    main_disease, main_conf = res['top_3'][0]
    
    st.markdown("<div class='section-header'>📊 Your Health Analysis Report</div>", unsafe_allow_html=True)
    
    # Main Prediction Card
    st.markdown(f"""
    <div class='big-card' style='text-align:center; background: linear-gradient(135deg, rgba(0, 217, 163, 0.1) 0%, rgba(0, 180, 216, 0.1) 100%);'>
        <p style='color: #B0B8C1; font-size: 16px; margin-bottom: 10px;'>Most Likely Condition</p>
        <h1 style='color:#00D9A3; font-size: 48px; margin: 20px 0; font-weight: 800;'>{main_disease}</h1>
        <p style='font-size: 20px; color: #E8EAED;'>Confidence Score: <strong>{main_conf:.1f}%</strong></p>
        <p style='color: #6B7280; font-size: 14px; margin-top: 15px;'>⏰ Analysis Date: {res['date']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Recommendations Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card' style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%); border-left: 4px solid #3B82F6;'>", unsafe_allow_html=True)
        st.markdown("### 🧪 Recommended Medical Tests")
        st.markdown(f"**{res['advice']['tests']}**")
        st.caption("Consult your doctor for these diagnostic tests")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card' style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%); border-left: 4px solid #10B981;'>", unsafe_allow_html=True)
        st.markdown("### 💊 Treatment & Care Suggestions")
        st.markdown(f"**{res['advice']['meds']}**")
        st.caption("Always follow your doctor's prescription")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-card' style='background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.15) 100%); border-left: 4px solid #EF4444;'>", unsafe_allow_html=True)
        st.markdown("### 🚨 Emergency Warning Signs")
        st.markdown(f"**Seek immediate medical help if you experience:**")
        st.markdown(f"• {res['advice']['emergency']}")
        st.caption("Don't delay - call emergency services immediately")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card' style='background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.1) 100%); border-left: 4px solid #F59E0B;'>", unsafe_allow_html=True)
        st.markdown("### 📋 Alternative Possibilities")
        if len(res['top_3']) > 1:
            for disease, confidence in res['top_3'][1:]:
                st.markdown(f"• **{disease}** ({confidence:.1f}% confidence)")
        else:
            st.markdown("No other significant possibilities detected")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Symptoms Summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='big-card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Reported Symptoms")
    symptoms_formatted = [s.replace("_", " ").title() for s in res['symptoms']]
    st.markdown(", ".join(symptoms_formatted))
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Download Report
    st.markdown("<br>", unsafe_allow_html=True)
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 40px; background: #f5f5f5; }}
            .container {{ background: white; padding: 40px; border-radius: 10px; max-width: 800px; margin: 0 auto; }}
            h1 {{ color: #00D9A3; border-bottom: 3px solid #00D9A3; padding-bottom: 10px; }}
            .section {{ margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }}
            .warning {{ color: #EF4444; font-weight: bold; padding: 15px; background: #FEE2E2; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>💉 MediAid AI - Medical Analysis Report</h1>
            <p><strong>Patient:</strong> {st.session_state.user_info[0]}</p>
            <p><strong>Report Date:</strong> {res['date']}</p>
            <hr>
            <div class="section">
                <h2>Primary Diagnosis: {main_disease}</h2>
                <p><strong>Confidence Level:</strong> {main_conf:.1f}%</p>
            </div>
            <div class="section">
                <h3>Symptoms Reported:</h3>
                <p>{', '.join(symptoms_formatted)}</p>
            </div>
            <div class="section">
                <h3>Recommended Tests:</h3>
                <p>{res['advice']['tests']}</p>
            </div>
            <div class="section">
                <h3>Treatment Recommendations:</h3>
                <p>{res['advice']['meds']}</p>
            </div>
            <div class="warning">
                <h3>⚠️ Emergency Warning Signs:</h3>
                <p>{res['advice']['emergency']}</p>
            </div>
            <hr>
            <p style="color: #6B7280; font-size: 12px; margin-top: 30px;">
                <em>This report is generated by MediAid AI and is intended for informational purposes only. 
                It does NOT replace professional medical advice, diagnosis, or treatment. 
                Always consult with qualified healthcare providers regarding any medical concerns.</em>
            </p>
        </div>
    </body>
    </html>
    """
    
    b64 = base64.b64encode(report_html.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="MediAid_Report_{main_disease}.html" style="display:inline-block; text-decoration:none; padding:14px 28px; background:linear-gradient(135deg, #00D9A3, #00B4D8); color:#000; border-radius:12px; font-weight:600; box-shadow: 0 4px 20px rgba(0, 217, 163, 0.25); transition: all 0.3s;">📥 Download Full Report</a>'
    st.markdown(f"<div style='text-align:center; margin: 30px 0;'>{href}</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_back1, col_back2, col_back3 = st.columns([1, 1, 1])
    with col_back2:
        if st.button("← Back to Dashboard", use_container_width=True):
            navigate_to("Dashboard")

def render_history():
    """Display user's medical history"""
    st.markdown("<div class='section-header'>📜 Your Medical History</div>", unsafe_allow_html=True)
    
    username = st.session_state.user_info[0]
    reports = run_query("SELECT * FROM reports WHERE username = ? ORDER BY id DESC", (username,), fetch_all=True)
    
    if not reports:
        st.info("📭 No medical reports found. Start your first symptom check to build your health history.")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🩺 Check Symptoms Now", use_container_width=True):
            navigate_to("SymptomCheck")
    else:
        st.success(f"✅ You have {len(reports)} medical report(s) in your history")
        st.markdown("<br>", unsafe_allow_html=True)
        
        for idx, report in enumerate(reports):
            # report: id, username, date, symptoms, prediction, confidence, recommendations
            with st.expander(f"📄 Report #{report[0]} - {report[2]} | Diagnosis: {report[4]} ({report[5]})", expanded=(idx==0)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Symptoms Reported:**")
                    symptoms = report[3].split(",")
                    symptoms_formatted = [s.strip().replace("_", " ").title() for s in symptoms]
                    st.write(", ".join(symptoms_formatted))
                
                with col2:
                    st.markdown("**Confidence:**")
                    st.markdown(f"### {report[5]}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                try:
                    recommendations = json.loads(report[6])
                    
                    rec_col1, rec_col2 = st.columns(2)
                    with rec_col1:
                        st.markdown("**Recommended Tests:**")
                        st.info(recommendations.get('tests', 'N/A'))
                    
                    with rec_col2:
                        st.markdown("**Treatment Suggestions:**")
                        st.success(recommendations.get('meds', 'N/A'))
                    
                    st.warning(f"⚠️ **Emergency Warning:** {recommendations.get('emergency', 'N/A')}")
                    
                except:
                    st.caption("Recommendation details unavailable")
                
                st.markdown("---")

def render_education():
    """Display health education articles"""
    st.markdown("<div class='section-header'>📚 Health Education Library</div>", unsafe_allow_html=True)
    
    st.info("💡 Browse through our curated collection of health articles and tips")
    
    articles = run_query("SELECT * FROM articles", fetch_all=True)
    
    if not articles:
        st.warning("No articles available at the moment")
    else:
        # Group by category
        categories = {}
        for article in articles:
            cat = article[3]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(article)
        
        # Display by category
        for category, cat_articles in categories.items():
            st.markdown(f"### 📑 {category}")
            
            for article in cat_articles:
                with st.expander(f"**{article[1]}**"):
                    st.markdown(article[2])
                    st.caption(f"Category: {article[3]}")
            
            st.markdown("<br>", unsafe_allow_html=True)

# --- ADMIN PAGES ---

def render_admin():
    """Admin panel for system management"""
    st.markdown("<div class='section-header'>⚙️ Admin Control Panel</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["👥 User Management", "📊 Analytics Dashboard", "🗄️ Database"])
    
    with tab1:
        st.markdown("### Registered Users")
        users = run_query("SELECT username, email, join_date, is_admin FROM users", fetch_all=True)
        
        if users:
            df_users = pd.DataFrame(users, columns=["Username", "Email", "Join Date", "Is Admin"])
            df_users['Is Admin'] = df_users['Is Admin'].apply(lambda x: "Yes" if x == 1 else "No")
            st.dataframe(df_users, use_container_width=True, hide_index=True)
        else:
            st.info("No users found")
    
    with tab2:
        st.markdown("### System Statistics")
        
        count_users = run_query("SELECT COUNT(*) FROM users", fetch_one=True)[0]
        count_reports = run_query("SELECT COUNT(*) FROM reports", fetch_one=True)[0]
        count_articles = run_query("SELECT COUNT(*) FROM articles", fetch_one=True)[0]
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Users", count_users, delta="Active")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Diagnoses", count_reports)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Health Articles", count_articles)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Recent activity
        st.markdown("### Recent Diagnostic Activity")
        recent_reports = run_query("SELECT username, date, prediction, confidence FROM reports ORDER BY id DESC LIMIT 10", fetch_all=True)
        
        if recent_reports:
            df_reports = pd.DataFrame(recent_reports, columns=["User", "Date", "Diagnosis", "Confidence"])
            st.dataframe(df_reports, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("### Database Management")
        st.info("🔧 Database management tools for administrators")
        
        col_db1, col_db2 = st.columns(2)
        
        with col_db1:
            st.markdown("**Database Status:**")
            st.success("✅ Connected and operational")
            st.caption(f"Database file: {DB_FILE}")
        
        with col_db2:
            st.markdown("**Quick Actions:**")
            if st.button("🔄 Refresh Statistics"):
                st.rerun()

# -----------------------------------------------------------------------------
# 6. MAIN ROUTER & NAVIGATION
# -----------------------------------------------------------------------------

def main():
    """Main application router"""
    
    # Sidebar Navigation
    if st.session_state.logged_in:
        user = st.session_state.user_info
        
        with st.sidebar:
            st.markdown(f"<h2 style='color: #00D9A3; text-align: center;'>👤 {user[0]}</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: #B0B8C1; font-size: 13px;'>{user[2]}</p>", unsafe_allow_html=True)
            
            if user[3] == 1:  # Is Admin
                st.success("🔑 **Admin Access**")
                if st.button("⚙️ Admin Panel", use_container_width=True):
                    navigate_to("Admin")
            
            st.markdown("---")
            st.markdown("### 🧭 Navigation")
            
            if st.button("🏠 Dashboard", use_container_width=True):
                navigate_to("Dashboard")
            if st.button("🩺 Symptom Checker", use_container_width=True):
                navigate_to("SymptomCheck")
            if st.button("👤 My Profile", use_container_width=True):
                navigate_to("Profile")
            if st.button("📜 Medical History", use_container_width=True):
                navigate_to("History")
            if st.button("📚 Health Articles", use_container_width=True):
                navigate_to("Education")
            
            st.markdown("---")
            
            if st.button("🚪 Logout", use_container_width=True):
                logout()
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.caption("💉 MediAid AI v2.0")
    
    # Page Routing Logic
    page = st.session_state.current_page
    
    # Public pages
    if page == "Landing":
        render_landing()
    elif page == "Login":
        render_login()
    elif page == "Signup":
        render_signup()
    
    # Redirect unauthorized users
    elif not st.session_state.logged_in:
        navigate_to("Landing")
    
    # Protected pages
    elif page == "Dashboard":
        render_dashboard()
    elif page == "Profile":
        render_profile()
    elif page == "SymptomCheck":
        render_symptom_check()
    elif page == "Results":
        render_results()
    elif page == "History":
        render_history()
    elif page == "Education":
        render_education()
    elif page == "Admin" and st.session_state.user_info[3] == 1:
        render_admin()
    else:
        render_dashboard()  # Fallback

    # Footer
    st.markdown("""
    <div class='footer'>
        <p>Developed with ❤️ by Team MediAid AI</p>
        <p style='margin-top: 5px;'>Awais • Urooj • Aqib • Suleman</p>
        <p style='margin-top: 10px; font-size: 11px;'>© 2024 MediAid AI. All rights reserved. | Privacy Policy | Terms of Service</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. APPLICATION ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()