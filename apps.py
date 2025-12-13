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

st.set_page_config(page_title="MediAid AI", page_icon="💉", layout="wide")

# Custom Dark Theme CSS
dark_css = """
<style>
.stApp {
    background: linear-gradient(180deg, #071019 0%, #0b1216 50%, #0f1720 100%);
    color: #E6EEF3;
}
.stButton>button {
    background: linear-gradient(90deg, #00BFA5, #00AFA0);
    color: black;
    border-radius: 8px;
    font-weight: bold;
    border: none;
    padding: 10px 20px;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #00AFA0, #009E90);
    color: white;
}
.big-card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}
.metric-card {
    background: rgba(0, 191, 165, 0.1);
    border-left: 4px solid #00BFA5;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 10px;
}
.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    background: -webkit-linear-gradient(#00BFA5, #E6EEF3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    color: #9CA3AF;
    font-size: 18px;
    margin-bottom: 30px;
}
.section-header {
    font-size: 24px;
    font-weight: bold;
    color: #00BFA5;
    margin-top: 20px;
    margin-bottom: 15px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 5px;
}
.footer {
    color: #6B7280;
    text-align: center;
    font-size: 12px;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
}
/* Success/Warning/Error boxes custom style */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 10px;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATABASE MANAGEMENT (SQLite)
# -----------------------------------------------------------------------------

DB_FILE = "mediaid.db"

def init_db():
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
                
    # Educational Articles (Admin managed in theory, seeded here)
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
        
        # Seed some articles
        articles = [
            ("Preventing Seasonal Flu", "Wash hands frequently, get vaccinated, and avoid close contact with sick people.", "Prevention"),
            ("Recognizing Dehydration", "Symptoms include dry mouth, fatigue, dark urine, and dizziness. Drink water immediately.", "General Health"),
            ("First Aid for Cuts", "Clean the wound with water, apply antibiotic ointment, and cover with a bandage.", "First Aid")
        ]
        c.executemany("INSERT INTO articles (title, content, category) VALUES (?, ?, ?)", articles)

    conn.commit()
    conn.close()

# Initialize DB on app start
init_db()

# DB Helper Functions
def run_query(query, params=(), fetch_one=False, fetch_all=False, commit=False):
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

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_user(username, password):
    hashed = make_hashes(password)
    user = run_query("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed), fetch_one=True)
    return user

def create_user(username, password, email):
    if run_query("SELECT * FROM users WHERE username = ?", (username,), fetch_one=True):
        return False
    hashed = make_hashes(password)
    run_query("INSERT INTO users (username, password, email, join_date) VALUES (?, ?, ?, ?)", 
              (username, hashed, email, str(datetime.date.today())), commit=True)
    return True

# -----------------------------------------------------------------------------
# 3. MODEL LOADING & LOGIC
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "Prototype_model", "medi_aid_disease_model.pkl")
LE_PATH = os.path.join(BASE_DIR, "Prototype_model", "medi_aid_label_encoders.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Prototype_model", "medi_aid_scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "Prototype_model", "feature_names.pkl")

# Load Resources
model, label_encoders, scaler, feature_names = None, None, None, None
try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
        # Handle joblib vs pickle differences
        if model is None: model = joblib.load(MODEL_PATH)

    if os.path.exists(LE_PATH):
        try: label_encoders = joblib.load(LE_PATH)
        except: 
            with open(LE_PATH, "rb") as f: label_encoders = pickle.load(f)

    if os.path.exists(SCALER_PATH):
        try: scaler = joblib.load(SCALER_PATH)
        except:
             with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)

    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH, "rb") as f: feature_names = pickle.load(f)
        
except Exception as e:
    st.error(f"Error loading AI models: {e}")

# Full expected feature list for input mapping
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
    st.session_state.user_info = None # (username, password, email, is_admin, date)
if "current_page" not in st.session_state:
    st.session_state.current_page = "Landing"

def navigate_to(page):
    st.session_state.current_page = page
    st.rerun()

def logout():
    st.session_state.logged_in = False
    st.session_state.user_info = None
    st.session_state.current_page = "Landing"
    st.rerun()

# -----------------------------------------------------------------------------
# 5. PAGE FUNCTIONS
# -----------------------------------------------------------------------------

# --- PUBLIC PAGES ---

def render_landing():
    st.markdown("<div class='title'>💉 MediAid AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Your Intelligent Health Companion</div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <p>MediAid AI uses advanced machine learning to analyze your symptoms and provide preliminary health insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔐 Login", use_container_width=True):
                navigate_to("Login")
        with col_b:
            if st.button("📝 Sign Up", use_container_width=True):
                navigate_to("Signup")

    st.markdown("---")
    st.warning("⚠️ **Medical Disclaimer:** This tool does not replace a doctor. In case of emergency, contact your local emergency services immediately.")

def render_login():
    st.markdown("<div class='section-header'>Login</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='big-card'>", unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Sign In"):
            user = verify_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_info = user
                navigate_to("Dashboard")
            else:
                st.error("Invalid credentials")
        if st.button("Don't have an account? Sign Up"):
            navigate_to("Signup")
        st.button("Back to Home", on_click=lambda: navigate_to("Landing"))
        st.markdown("</div>", unsafe_allow_html=True)

def render_signup():
    st.markdown("<div class='section-header'>Create Account</div>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='big-card'>", unsafe_allow_html=True)
        new_user = st.text_input("Choose Username")
        new_email = st.text_input("Email Address")
        new_pass = st.text_input("Password", type="password")
        confirm_pass = st.text_input("Confirm Password", type="password")
        
        if st.checkbox("I agree to the Medical Disclaimer and Terms of Service"):
            if st.button("Register"):
                if new_pass != confirm_pass:
                    st.error("Passwords do not match")
                elif not new_user or not new_pass:
                    st.error("All fields are required")
                else:
                    if create_user(new_user, new_pass, new_email):
                        st.success("Account created! Please login.")
                        navigate_to("Login")
                    else:
                        st.error("Username already exists")
        
        st.button("Back to Login", on_click=lambda: navigate_to("Login"))
        st.markdown("</div>", unsafe_allow_html=True)

# --- USER PRIVATE PAGES ---

def render_dashboard():
    user = st.session_state.user_info[0]
    st.markdown(f"<div class='title'>Welcome, {user}</div>", unsafe_allow_html=True)
    
    # 4 Main Actions
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("🏥 **Start Diagnosis**")
        if st.button("Check Symptoms", use_container_width=True):
            navigate_to("SymptomCheck")
    with c2:
        st.success("👤 **Medical Profile**")
        if st.button("Update Profile", use_container_width=True):
            navigate_to("Profile")
    with c3:
        st.warning("📜 **History**")
        if st.button("View Reports", use_container_width=True):
            navigate_to("History")
    with c4:
        st.error("📚 **Education**")
        if st.button("Health Articles", use_container_width=True):
            navigate_to("Education")
    
    # Quick Tips Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='big-card'>", unsafe_allow_html=True)
    st.subheader("💡 Daily Health Tip")
    st.write("Stay hydrated! Drinking enough water helps regulate body temperature and keeps joints lubricated.")
    st.markdown("</div>", unsafe_allow_html=True)

def render_profile():
    st.markdown("<div class='section-header'>Medical Profile</div>", unsafe_allow_html=True)
    username = st.session_state.user_info[0]
    
    # Load existing data
    data = run_query("SELECT * FROM profiles WHERE username = ?", (username,), fetch_one=True)
    # Default values
    age, gender, height, weight, bg, allergies, chronic, meds, surgs = (
        25, "Male", 170.0, 70.0, "O+", "None", "None", "None", "None"
    )
    if data:
        # data[0] is username
        age, gender, height, weight, bg, allergies, chronic, meds, surgs = data[1:]

    with st.form("profile_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 1, 120, int(age))
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(gender) if gender in ["Male", "Female", "Other"] else 0)
            height = st.number_input("Height (cm)", 50.0, 250.0, float(height))
            weight = st.number_input("Weight (kg)", 20.0, 200.0, float(weight))
            blood_group = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], index=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"].index(bg) if bg in ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"] else 6)
        with c2:
            allergies = st.text_area("Known Allergies", allergies)
            chronic_diseases = st.text_area("Chronic Conditions (Diabetes, BP, etc.)", chronic)
            medications = st.text_area("Current Medications", meds)
            surgeries = st.text_area("Past Surgeries", surgs)
        
        if st.form_submit_button("Save Profile"):
            # Upsert
            run_query("REPLACE INTO profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                      (username, age, gender, height, weight, blood_group, allergies, chronic_diseases, medications, surgeries), commit=True)
            st.success("Profile updated successfully!")

def render_symptom_check():
    st.markdown("<div class='section-header'>Symptom Checker</div>", unsafe_allow_html=True)
    
    if model is None:
        st.error("Model files are missing. Cannot proceed.")
        return

    # 1. Basic Info (Pre-filled from profile if available)
    username = st.session_state.user_info[0]
    prof = run_query("SELECT * FROM profiles WHERE username = ?", (username,), fetch_one=True)
    
    default_age = prof[1] if prof else 25
    default_gender = prof[2] if prof else "Male"
    
    with st.expander("Step 1: Patient Details", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 1, 100, int(default_age))
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(default_gender))
        with c3:
            region = st.selectbox("Region", ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan", "Islamabad", "Gilgit-Baltistan", "Azad Kashmir"])
        
        c4, c5 = st.columns(2)
        with c4:
            duration_days = st.number_input("How long have you had symptoms? (Days)", 1, 30, 3)
        with c5:
            comorbidity = st.selectbox("Any existing condition?", ["None", "Diabetes", "Hypertension", "Chronic Lung Disease", "HIV", "Heart Disease"])

    # 2. Symptoms Selection
    st.subheader("Step 2: Select Symptoms")
    
    symptom_list = EXPECTED_FEATURES[5:] # Skip demographic cols
    
    # Categorize (Simple grouping for UI)
    categories = {
        "General": ['fever', 'fatigue', 'chills', 'sweating', 'dehydration', 'weight_loss', 'loss_of_appetite'],
        "Pain": ['headache', 'joint_pain', 'muscle_pain', 'back_pain', 'chest_pain', 'abdominal_pain', 'lower_abdominal_pain', 'retro_orbital_pain'],
        "Respiratory/ENT": ['cough', 'persistent_cough', 'sore_throat', 'runny_nose', 'shortness_of_breath', 'sputum', 'loss_of_smell_taste'],
        "Digestive": ['nausea', 'vomiting', 'diarrhea', 'constipation', 'jaundice'],
        "Skin/Other": ['rash', 'itching', 'vesicular_rash', 'dark_urine', 'dysuria', 'frequency', 'conjunctivitis', 'night_sweats']
    }
    
    selected_symptoms = []
    
    tabs = st.tabs(categories.keys())
    
    for i, (cat_name, cat_syms) in enumerate(categories.items()):
        with tabs[i]:
            cols = st.columns(3)
            for j, sym in enumerate(cat_syms):
                if sym in symptom_list:
                    # Checkbox for presence
                    is_present = cols[j%3].checkbox(sym.replace("_", " ").title(), key=f"chk_{sym}")
                    if is_present:
                        # Optional: Severity slider (Not used by model yet, but good for record)
                        severity = cols[j%3].slider(f"Severity ({sym})", 1, 10, 5, key=f"sev_{sym}", label_visibility="collapsed")
                        selected_symptoms.append(sym)

    if st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            process_prediction(age, gender, region, duration_days, comorbidity, selected_symptoms)

def process_prediction(age, gender, region, duration_days, comorbidity, selected_symptoms):
    # 1. Prepare Dataframe
    row = {s: (1 if s in selected_symptoms else 0) for s in EXPECTED_FEATURES if s not in ['age','gender','region','duration_days','comorbidity']}
    row.update({'age': age, 'gender': gender, 'region': region, 'duration_days': duration_days, 'comorbidity': comorbidity})
    
    df_input = pd.DataFrame([row])
    
    # 2. Preprocessing (Encoding/Scaling)
    # Ensure all columns exist
    for col in EXPECTED_FEATURES:
        if col not in df_input.columns: df_input[col] = 0
    df_input = df_input[EXPECTED_FEATURES] # Enforce order
    
    # Encode
    for col in ['gender', 'region', 'comorbidity']:
        if label_encoders and col in label_encoders:
            try: df_input[col] = label_encoders[col].transform(df_input[col])
            except: df_input[col] = 0
        else: df_input[col] = 0
            
    # Scale
    if scaler:
        try: df_input[['age', 'duration_days']] = scaler.transform(df_input[['age', 'duration_days']])
        except: pass

    # 3. Predict
    prediction_cls = model.predict(df_input.to_numpy().reshape(1, -1))[0]
    
    # Try to get probabilities
    top_3 = []
    try:
        probs = model.predict_proba(df_input.to_numpy().reshape(1, -1))[0]
        classes = model.classes_
        # Sort and get top 3
        sorted_indices = np.argsort(probs)[::-1]
        for i in range(3):
            if i < len(classes):
                idx = sorted_indices[i]
                top_3.append((classes[idx], probs[idx] * 100))
    except:
        # Fallback if model doesn't support probability
        top_3.append((prediction_cls, 100.0))

    # 4. Save result to Session State and DB
    result_data = {
        "top_3": top_3,
        "symptoms": selected_symptoms,
        "date": str(datetime.datetime.now()),
        "advice": get_medical_advice(top_3[0][0])
    }
    
    # Save to DB
    run_query("INSERT INTO reports (username, date, symptoms, prediction, confidence, recommendations) VALUES (?, ?, ?, ?, ?, ?)", 
              (st.session_state.user_info[0], result_data['date'], ",".join(selected_symptoms), 
               top_3[0][0], f"{top_3[0][1]:.1f}%", json.dumps(result_data['advice'])), commit=True)
    
    st.session_state.last_result = result_data
    navigate_to("Results")

def get_medical_advice(disease):
    # Dictionary of advice
    data = {
        "Dengue": {"tests": "NS1 Antigen, CBC (Platelets)", "meds": "Paracetamol, Hydration", "emergency": "Bleeding gums, severe abdominal pain"},
        "Malaria": {"tests": "Blood Smear, RDT", "meds": "Antimalarials", "emergency": "Cerebral symptoms, jaundice"},
        "Typhoid": {"tests": "Widal, Blood Culture", "meds": "Antibiotics", "emergency": "Intestinal perforation symptoms"},
        "COVID-19": {"tests": "PCR, Antigen", "meds": "Isolation, Supportive", "emergency": "Low oxygen, chest pressure"},
        "Pneumonia": {"tests": "Chest X-Ray", "meds": "Antibiotics", "emergency": "Difficulty breathing, blue lips"},
    }
    return data.get(disease, {"tests": "Consult Doctor", "meds": "Consult Doctor", "emergency": "Severe Worsening"})

def render_results():
    if "last_result" not in st.session_state:
        navigate_to("Dashboard")
        return
        
    res = st.session_state.last_result
    st.markdown("<div class='section-header'>Analysis Report</div>", unsafe_allow_html=True)
    
    # Top Prediction
    main_disease, main_conf = res['top_3'][0]
    
    st.markdown(f"""
    <div class='big-card' style='text-align:center;'>
        <h2>Most Likely Condition</h2>
        <h1 style='color:#00BFA5; font-size: 50px;'>{main_disease}</h1>
        <p>Confidence: {main_conf:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed Analysis
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("### 🧪 Recommended Tests")
        st.write(res['advice']['tests'])
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("### 💊 Suggested OTC / Home Care")
        st.write(res['advice']['meds'])
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='metric-card' style='border-left: 4px solid #FF4B4B; background: rgba(255, 75, 75, 0.1);'>", unsafe_allow_html=True)
        st.write("### 🚨 Emergency Warning")
        st.write(f"Seek immediate help if: {res['advice']['emergency']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.write("### Other Possibilities")
        for d, c in res['top_3'][1:]:
            st.write(f"- **{d}** ({c:.1f}%)")

    # Download Report Button (HTML generation)
    report_html = f"""
    <h1>MediAid AI Medical Report</h1>
    <p><strong>Patient:</strong> {st.session_state.user_info[0]}</p>
    <p><strong>Date:</strong> {res['date']}</p>
    <hr>
    <h2>Prediction: {main_disease} ({main_conf:.1f}%)</h2>
    <h3>Symptoms Reported</h3>
    <p>{', '.join(res['symptoms'])}</p>
    <h3>Recommendations</h3>
    <ul>
        <li><strong>Tests:</strong> {res['advice']['tests']}</li>
        <li><strong>Medicines:</strong> {res['advice']['meds']}</li>
    </ul>
    <p style='color:red'><strong>Warning:</strong> {res['advice']['emergency']}</p>
    <hr>
    <p><em>Generated by MediAid AI. Not a substitute for professional medical advice.</em></p>
    """
    b64 = base64.b64encode(report_html.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="MediAid_Report.html" style="text-decoration:none; padding:10px 20px; background:#00BFA5; color:white; border-radius:5px;">📥 Download Report</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Back to Dashboard", on_click=lambda: navigate_to("Dashboard"))

def render_history():
    st.markdown("<div class='section-header'>My Reports History</div>", unsafe_allow_html=True)
    username = st.session_state.user_info[0]
    reports = run_query("SELECT * FROM reports WHERE username = ? ORDER BY id DESC", (username,), fetch_all=True)
    
    if not reports:
        st.info("No reports found.")
    else:
        for r in reports:
            # r: id, user, date, syms, pred, conf, recs
            with st.expander(f"{r[2]} - {r[4]} ({r[5]})"):
                st.write(f"**Symptoms:** {r[3]}")
                st.write(f"**Recommendation:** {json.loads(r[6])['meds']}")

def render_education():
    st.markdown("<div class='section-header'>Health Articles</div>", unsafe_allow_html=True)
    articles = run_query("SELECT * FROM articles", fetch_all=True)
    for a in articles:
        st.markdown(f"### {a[1]}") # Title
        st.info(f"Category: {a[3]}")
        st.write(a[2]) # Content
        st.markdown("---")

# --- ADMIN PAGES ---

def render_admin():
    st.markdown("<div class='section-header'>Admin Panel</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["User Management", "Analytics", "Database"])
    
    with tab1:
        users = run_query("SELECT username, email, join_date FROM users", fetch_all=True)
        st.dataframe(pd.DataFrame(users, columns=["Username", "Email", "Join Date"]))
    
    with tab2:
        count_users = run_query("SELECT COUNT(*) FROM users", fetch_one=True)[0]
        count_reports = run_query("SELECT COUNT(*) FROM reports", fetch_one=True)[0]
        
        k1, k2 = st.columns(2)
        k1.metric("Total Users", count_users)
        k2.metric("Total Diagnostics Run", count_reports)
        
    with tab3:
        st.write("Manage Diseases/Articles here (Placeholder for future expansion)")

# -----------------------------------------------------------------------------
# 6. MAIN ROUTER
# -----------------------------------------------------------------------------

def main():
    # Sidebar
    if st.session_state.logged_in:
        user = st.session_state.user_info
        st.sidebar.title(f"👤 {user[0]}")
        if user[3] == 1: # Is Admin
            st.sidebar.success("Admin Mode Active")
            if st.sidebar.button("Admin Panel"): navigate_to("Admin")
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🏠 Dashboard"): navigate_to("Dashboard")
        if st.sidebar.button("🩺 Check Symptoms"): navigate_to("SymptomCheck")
        if st.sidebar.button("📝 Profile"): navigate_to("Profile")
        if st.sidebar.button("📜 History"): navigate_to("History")
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout"): logout()
    
    # Routing
    page = st.session_state.current_page
    
    if page == "Landing": render_landing()
    elif page == "Login": render_login()
    elif page == "Signup": render_signup()
    
    elif not st.session_state.logged_in:
        # Redirect unauthorized access
        navigate_to("Landing")
        
    elif page == "Dashboard": render_dashboard()
    elif page == "Profile": render_profile()
    elif page == "SymptomCheck": render_symptom_check()
    elif page == "Results": render_results()
    elif page == "History": render_history()
    elif page == "Education": render_education()
    elif page == "Admin" and st.session_state.user_info[3] == 1: render_admin()
    else: render_dashboard() # Fallback

    # Footer
    st.markdown("<div class='footer'>Developed by Team MediAid AI — Awais • Urooj • Aqib • Suleman</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()