import streamlit as st
from ml import predict_with_model

# --- Simulated user store (in-memory) ---
# Store users in a persistent variable across reruns
if "users" not in st.session_state:
    st.session_state["users"] = st.session_state.get("users", {"admin": "admin123"})

# --- Session login state ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- Login/signup form ---
def login_form():
    st.title("🔐 Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.logged_in = True
                st.success("✅ Logged in successfully.")
            else:
                st.error("❌ Invalid username or password.")

    with tab2:
        new_user = st.text_input("New Username", key="signup_user")
        new_pass = st.text_input("New Password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            if new_user in st.session_state.users:
                st.warning("⚠️ Username already exists.")
            elif new_user and new_pass:
                st.session_state.users[new_user] = new_pass
                st.success("✅ Account created! Please log in.")
            else:
                st.warning("⚠️ Please enter both username and password.")

# --- Main app ---
def main_app():
    st.title("🦠 Covid-19 Symptoms Checker")

    # SYMPTOMS (optional)
    symptom_options = [
        "Fever", "Tiredness", "Dry-Cough", "Difficulty-in-Breathing", 
        "Sore-Throat", "Pains", "Nasal-Congestion", "Runny-Nose", "Diarrhea"
    ]
    symptoms = st.multiselect("Select your symptoms (if any):", symptom_options)

    # Age group (no default)
    age_options = ["0-9", "10-19", "20-24", "25-59", "60+"]
    age = st.radio("Select your age group:", age_options, index=None)

    # Contact (no default)
    contact_options = ["Yes", "No", "Maybe"]
    contact = st.radio("Had contact with a COVID-positive person?", contact_options, index=None)

    # Model
    model_options = ["Naive Bayes", "Decision Tree", "SVM", "KNN", "Random Forest", "Logistic Regression"]
    model_name = st.selectbox("Choose ML Model:", [""] + model_options)
    model_name = model_name if model_name != "" else None

    if st.button("Check"):
        if age is None or contact is None or model_name is None:
            st.warning("⚠️ Please fill out all details.")
        else:
            prediction = predict_with_model(symptoms, age, contact, model_name)
            if prediction == 1:
                st.error("⚠️ You are likely infected with COVID-19.")
            else:
                st.success("✅ You are unlikely to be infected with COVID-19.")

# --- App Router ---
if not st.session_state.logged_in:
    login_form()
else:
    main_app()
