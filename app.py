import json, io, base64, pandas as pd
import streamlit as st
import joblib   # or pickle if you prefer

# ---------- Page & global config ----------
st.set_page_config(
    page_title="Gut‑Heart Risk Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject minimal CSS (optional)
st.markdown(
    """
    <style>
        .reportview-container { background: #f9f9f9; }
        .risk-high {color:#d62728;font-weight:700;}
        .risk-low  {color:#2ca02c;font-weight:700;}
        .advice-card {background:#272731;border-radius:8px;padding:1rem;margin-bottom:0.75rem;box-shadow:0 1px 4px rgba(0,0,0,0.05);}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Load assets ----------
@st.cache_resource
def load_assets():
    rf = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("microbe_knowledge.json") as f:
        kb = json.load(f)
    return rf, scaler, kb


rf_model, scaler, microbe_kb = load_assets()

model_dict = {"Random Forest": rf_model}

def top_feature_importance(model, k=15):
    """Return a DataFrame of the k most important features."""
    imp = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    imp = imp.sort_values(ascending=False).head(k)
    return imp.reset_index().rename(columns={"index": "Microbe", 0: "Importance"})


# ---------- Sidebar ----------
st.sidebar.image("logo.png", width=180)
page = st.sidebar.radio("Navigate", ["🏠 Home", "📁 Upload & Predict", "❓ FAQ"])
# chosen_model_name = st.sidebar.selectbox("Choose model", list(model_dict.keys()))
# chosen_model = model_dict[chosen_model_name]
chosen_model = rf_model
st.sidebar.markdown("---")
st.sidebar.caption("Gut‑Heart | For research use only")

# ---------- Helper functions ----------
def predict(df: pd.DataFrame):
    X = df[chosen_model.feature_names_in_]
    X_scaled = scaler.transform(X)
    prob = chosen_model.predict_proba(X_scaled)[0,1]
    # prob = chosen_model.predict_proba(df)[0, 1]
    label = "High Risk" if prob >= 0.5 else "Low Risk"
    return prob, label

def advice_for_microbe(raw_name: str):
    """Return the KB entry whose aliases contain `raw_name` (after cleanup)."""
    # quick cleanup → turn taxonomy paths into simple species names
    if "s__" in raw_name:
        raw_name = raw_name.split("s__")[-1]
    if "|" in raw_name:
        raw_name = raw_name.split("|")[-1]
    canonical = raw_name.replace("_", " ").strip().lower()

    for entry in microbe_kb:
        if any(canonical == alias.lower().replace("_", " ") for alias in entry["aliases"]):
            return entry
    return None

def sample_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="sample_microbiome.csv">📥 Download sample CSV</a>'

# ---------- Home ----------
if page == "🏠 Home":
    st.title("🧬 Gut‑Microbiome Cardiovascular Risk Predictor")
    st.write(
        """
        Upload relative‑abundance microbiome data (one sample per row) 
        and receive an instant cardiovascular‑risk estimate, an explanation 
        of key microbial drivers, and personalised diet/lifestyle guidance.
        """
    )
    st.markdown(sample_csv_download_link(
        pd.read_csv("jiez_preprocessed_with_age.csv").head(3)
    ), unsafe_allow_html=True)
    st.image("hero_illustration.png")

# ---------- Upload & Predict ----------
elif page == "📁 Upload & Predict":
    st.header("Step 1 — Upload your microbiome CSV")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded:
        user_df = pd.read_csv(uploaded)
        st.success("File loaded successfully! Showing preview:")
        st.dataframe(user_df.head())

        # --- Prediction ---
        st.header("Step 2 — Model Prediction")
        prob, label = predict(user_df)
        st.metric("CVD Risk Probability", f"{prob:.2%}", delta=None)
        st.markdown(
            f"### Result: **<span class='risk-{'high' if label=='High Risk' else 'low'}'>{label}</span>**",
            unsafe_allow_html=True
        )

        # --- Feature‑importance explanation ---
        st.subheader("Top contributing microbes")
        imp_df = top_feature_importance(chosen_model, k=15)
        st.bar_chart(imp_df.set_index("Microbe"))

        # --- Lifestyle advice ---
        top_microbes = imp_df["Microbe"].tolist()
        st.header("Lifestyle & Diet Suggestions for Top‑Impact Microbes")
        for microbe in top_microbes[:5]:
            advice = advice_for_microbe(microbe)
            if advice:
                with st.container():
                    st.markdown(
                        f"""
                        <div class="advice-card">
                        <h4>{microbe} ({'🔺' if advice['effect']=='risk_increase' else '🔻'} {advice['effect'].replace('_',' ')})</h4>
                        <p><em>Evidence:</em> {advice['evidence']}</p>
                        <p><strong>Try:</strong></p>
                        <ul>{"".join(f"<li>{i}</li>" for i in advice['interventions'])}</ul>
                        <p><strong>Indian food ideas:</strong> {", ".join(advice['food_examples'])}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


        # --- Disclaimer ---
        st.info(
            "⚠️ **Research‑grade prototype** – not intended for clinical diagnosis. "
            "Consult a healthcare professional before making health decisions."
        )

# ---------- FAQ ----------
else:
    st.header("Frequently Asked Questions")
    with st.expander("What format should my CSV be in?"):
        st.write(
            """
            • Rows = samples  
            • Columns = microbial relative abundances (k__… species)  
            • Include *exactly* the same feature names used during training  
            • Optional metadata columns are ignored at prediction time
            """
        )
    with st.expander("Why does SHAP not work?"):
        st.write(
            "The current tree‑SHAP implementation can’t handle the high‑dimensional "
            "sparse feature matrix from microbiome data."
        )
    with st.expander("How were the lifestyle suggestions built?"):
        st.write(
            "They come from a manually curated JSON knowledge base that maps each microbe "
            "to peer‑reviewed evidence, dietary interventions, and India‑specific food examples."
        )
