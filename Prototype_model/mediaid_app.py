# mediaid_app.py
# MediAid AI - Streamlit app (Dark mode + Login/Signup + SHAP + Feature alignment)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hashlib
import os

try:
    import shap
    from streamlit_shap import st_shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# -------------------------
# Helper functions
# -------------------------
USERS_FILE = "users.csv"

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        return pd.read_csv(USERS_FILE)
    else:
        return pd.DataFrame(columns=["username", "password"])

def save_users(df):
    df.to_csv(USERS_FILE, index=False)

def verify_credentials(username, password):
    users = load_users()
    hashed = make_hashes(password)
    match = users[(users["username"] == username) & (users["password"] == hashed)]
    return not match.empty


# -------------------------
# Session state
# -------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None


# -------------------------
# Dark Mode Styling
# -------------------------
st.set_page_config(page_title="MediAid AI", page_icon="💉", layout="wide")

dark_css = """
<style>
.stApp {
    background: linear-gradient(180deg, #071019 0%, #0b1216 50%, #0f1720 100%);
    color: #E6EEF3;
}
.stButton>button {
    background: linear-gradient(90deg, #00BFA5, #00AFA0);
    color: black;
    border-radius: 10px;
    padding: 8px 12px;
}
.big-card {
    background: rgba(255,255,255,0.03);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 6px 18px rgba(2,6,23,0.6);
}
.title {
    font-size: 38px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 6px;
}
.subtitle {
    text-align: center;
    color: #9CA3AF;
    margin-bottom: 20px;
}
.footer {
    color: #9CA3AF;
    text-align: center;
    font-size: 13px;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)


# -------------------------
# Load Model and Preprocessors
# -------------------------
MODEL_PATH = 'C:\\Users\\pc\\Desktop\\MediAid_Ai\\Prototype_model\\medi_aid_disease_model.pkl'
LE_PATH = 'C:\\Users\\pc\\Desktop\\MediAid_Ai\\Prototype_model\\medi_aid_label_encoders.pkl'
SCALER_PATH = 'C:\\Users\\pc\\Desktop\\MediAid_Ai\\Prototype_model\\medi_aid_scaler.pkl'

model = None
label_encoders = None
scaler = None
model_missing = False

if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH) and os.path.exists(SCALER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(LE_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.error("Error loading model files: " + str(e))
        model_missing = True
else:
    model_missing = True

FEATURES_PATH = 'C:\\Users\\pc\\Desktop\\MediAid_Ai\\Prototype_model\\feature_names.pkl'

if os.path.exists(FEATURES_PATH):
    expected_features = joblib.load(FEATURES_PATH)
else:
    st.error("feature_names.pkl not found. Please add it to the project folder.")
    st.stop()


# -------------------------
# Login / Signup Screen
# -------------------------
if not st.session_state.logged_in:
    st.markdown("<div class='title'>💉 MediAid AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Sign in to access the MediAid AI predictor</div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("<div class='big-card'>", unsafe_allow_html=True)
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if verify_credentials(username, password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.rerun()

            else:
                st.error("Invalid username or password.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='big-card'>", unsafe_allow_html=True)
        st.subheader("🆕 Sign Up")
        new_user = st.text_input("Choose a username")
        new_pass = st.text_input("Choose a password", type="password")
        if st.button("Sign Up"):
            if not new_user or not new_pass:
                st.warning("Please provide both username and password.")
            else:
                users = load_users()
                if new_user in list(users["username"]):
                    st.warning("Username already exists.")
                else:
                    hashed = make_hashes(new_pass)
                    new_entry = pd.DataFrame([[new_user, hashed]], columns=["username", "password"])
                    users = pd.concat([users, new_entry], ignore_index=True)
                    save_users(users)
                    st.success("Account created successfully!")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><div class='footer'>Developed by Team MediAid AI — Awais • Urooj • Aqib • Suleman</div>", unsafe_allow_html=True)


# -------------------------
# Main App (After Login)
# -------------------------
else:
    st.sidebar.markdown(f"**Logged in as:** `{st.session_state.user}`")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.experimental_rerun()

    st.markdown("<div class='title'>💉 MediAid AI — Disease Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-driven health insights and explainability</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🏥 Prediction", "👨‍⚕️ About Us"])

    # -------------------------
    # Prediction Tab
    # -------------------------
    with tab1:
        st.markdown("<div class='big-card'>", unsafe_allow_html=True)
        st.header("🩺 Symptom-Based Prediction")

        if model_missing:
            st.error("Model or preprocessing files are missing. Please ensure all .pkl files are in this folder.")
            st.stop()

        # Sidebar inputs
        st.sidebar.header("Patient Info")
        age = st.sidebar.slider("Age", 1, 100, 25)
        gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
        region = st.sidebar.selectbox("Region", ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan", "Islamabad", "Gilgit-Baltistan", "Azad Kashmir"])
        duration_days = st.sidebar.slider("Duration of symptoms (days)", 1, 30, 5)
        comorbidity = st.sidebar.selectbox("Comorbidity", ["None","Diabetes","Hypertension","Chronic Lung Disease","HIV","Heart Disease"])

        # Full expected feature list
        expected_features = [
            'age','gender','region','duration_days','comorbidity',
            'abdominal_pain','back_pain','chest_pain','chills','conjunctivitis','constipation','cough','dark_urine',
            'dehydration','diarrhea','dysuria','fatigue','fever','frequency','headache','itching','joint_pain',
            'jaundice','loss_of_appetite','loss_of_smell_taste','lower_abdominal_pain','muscle_pain','nausea',
            'night_sweats','persistent_cough','rash','retro_orbital_pain','runny_nose','shortness_of_breath',
            'sore_throat','sputum','sweating','vesicular_rash','vomiting','weight_loss'
        ]

        st.subheader("Select Symptoms")
        selected_symptoms = []
        cols = st.columns(3)
        for i, sym in enumerate(expected_features[5:]):
            if cols[i % 3].checkbox(sym.replace('_', ' ').capitalize()):
                selected_symptoms.append(sym)

        if st.button("🔍 Predict"):
            if not selected_symptoms:
                st.warning("Please select at least one symptom.")
            else:
                # Build safe input DataFrame
                row = {s: (1 if s in selected_symptoms else 0) for s in expected_features if s not in ['age','gender','region','duration_days','comorbidity']}
                row.update({'age': age, 'gender': gender, 'region': region, 'duration_days': duration_days, 'comorbidity': comorbidity})
                df_input = pd.DataFrame([row])

                # Ensure all expected columns exist and in order
                for col in expected_features:
                    if col not in df_input.columns:
                        df_input[col] = 0
                df_input = df_input[expected_features]

                # Encode categorical fields safely
                for col in ['gender','region','comorbidity']:
                    le = label_encoders.get(col, None)
                    if le is not None:
                        try:
                            df_input[col] = le.transform(df_input[col])
                        except Exception:
                            df_input[col] = 0
                    else:
                        df_input[col] = 0

                # Scale numerical safely
                if scaler is not None:
                    try:
                        df_input[['age','duration_days']] = scaler.transform(df_input[['age','duration_days']])
                    except Exception as e:
                        st.warning(f"Scaler issue: {e}")
                # Ensure df_input matches the exact order of training features
                missing_cols = [col for col in expected_features if col not in df_input.columns]
                extra_cols = [col for col in df_input.columns if col not in expected_features]

                # Add missing columns as 0
                for col in missing_cols:
                    df_input[col] = 0

                # Drop unexpected extra columns
                df_input = df_input[[c for c in df_input.columns if c in expected_features]]

                # Finally, reorder columns exactly as the model was trained
                df_input = df_input[expected_features]

                # Predict
                #prediction = model.predict(df_input)[0]
                prediction = model.predict(df_input.to_numpy().reshape(1, -1))

                st.success(f"### 🧠 Predicted Disease: {prediction}")

                # Recommendations
                recommended_tests = {
                    "Dengue": "NS1 antigen, CBC (platelets)",
                    "Malaria": "RDT, Blood smear",
                    "Typhoid": "Widal test, Blood culture",
                    "Tuberculosis": "Chest X-ray, Sputum AFB",
                    "Hepatitis A": "ALT/AST, HAV IgM",
                    "Hepatitis C": "HCV Antibody, HCV RNA",
                    "Pneumonia": "Chest X-ray, CBC",
                    "COVID-19": "RT-PCR, Antigen test",
                    "Urinary_Tract_Infection": "Urine analysis, Urine culture",
                    "Gastroenteritis": "Stool routine, Electrolytes",
                    "Measles": "IgM antibody test",
                    "Chickenpox": "Clinical evaluation or VZV IgM"
                }
                recommended_medicines = {
                    "Dengue": "Paracetamol, hydration",
                    "Malaria": "Antimalarial therapy, antipyretic",
                    "Typhoid": "Azithromycin, ciprofloxacin",
                    "Tuberculosis": "Refer to TB clinic (standard regimen)",
                    "Hepatitis A": "Supportive care, hydration",
                    "Hepatitis C": "Refer for antiviral therapy",
                    "Pneumonia": "Antibiotics, antipyretic",
                    "COVID-19": "Supportive care, isolation",
                    "Urinary_Tract_Infection": "Nitrofurantoin, hydration",
                    "Gastroenteritis": "ORS, antidiarrheal (as advised)",
                    "Measles": "Vitamin A, supportive care",
                    "Chickenpox": "Antihistamines, calamine lotion"
                }

                st.write("### 🧾 Recommended Tests:")
                #st.info(recommended_tests.get(prediction, "Consult a healthcare provider."))
                st.info(recommended_tests.get(prediction[0], "Consult a healthcare provider."))

                st.write("### 💊 Recommended Medicines:")
                st.info(recommended_medicines.get(prediction[0], "Consult a healthcare provider."))

  # SHAP Explain
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_input)

    # Get predicted label and index
    pred_label = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
    pred_label = pred_label.item() if hasattr(pred_label, "item") else pred_label
    idx = int(np.where(np.array(model.classes_) == pred_label)[0][0])

    # Handle different shap shapes
    if isinstance(shap_values, list):
        # multiclass: list of arrays
        class_shap = shap_values[idx][0]
    elif shap_values.ndim == 3:
        # multiclass: single 3D array (1, features, classes)
        class_shap = shap_values[0, :, idx]
    elif shap_values.ndim == 2:
        # binary/multiclass single sample
        class_shap = shap_values[0]
    else:
        raise ValueError(f"Unexpected shap array dims: {shap_values.ndim}")

    # Get expected value
    expected = explainer.expected_value
    if isinstance(expected, (list, tuple, np.ndarray)):
        ev = expected[idx] if len(expected) > 1 else expected[0]
    else:
        ev = expected

    # Create impact DataFrame
    impact = np.abs(class_shap).flatten()
    imp_df = (
        pd.DataFrame({'feature': df_input.columns, 'impact': impact})
        .sort_values('impact', ascending=False)
        .head(8)
    )

    st.write("Top Influential Symptoms:")
    st.bar_chart(imp_df.set_index('feature'))

    st.write("Detailed SHAP Force Plot:")
    st_shap(shap.force_plot(ev, class_shap, df_input))

except Exception as e:
    st.error(f"Error computing SHAP: {e}")
    try:
        sv_shape = (
            [arr.shape for arr in shap_values] if isinstance(shap_values, list) else shap_values.shape
        )
    except Exception:
        sv_shape = "unknown"

    st.write("Debug info:")
    st.write(f"model.classes_: {model.classes_}")
    st.write(f"pred_label: {pred_label}")
    st.write(f"index found (idx): {locals().get('idx', 'not available')}")
    st.write(f"shap_values shape(s): {sv_shape}")
    st.write(f"df_input.shape: {df_input.shape}")


    # -------------------------
    # About Us Tab
    # -------------------------
    with tab2:
        st.markdown("<div class='big-card'>", unsafe_allow_html=True)
        st.header("👨‍⚕️ About Team MediAid AI")
        st.markdown("""
        We are a team of Software Engineering students passionate about leveraging AI for healthcare innovation.  
        MediAid AI aims to bring early disease awareness through intelligent prediction.

        **Team Members**  
        🧑‍💻 Awais  
        👩‍💻 Urooj  
        👨‍💻 Aqib  
        👨‍💻 Suleman  
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='footer'>© Team MediAid AI • Educational prototype • Not a substitute for medical advice</div>", unsafe_allow_html=True)
