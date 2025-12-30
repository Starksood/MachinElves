import os

# --- MAC OS & STREAMLIT CRASH FIX ---
# These MUST go before any other imports!
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU to avoid Metal/GPU conflicts

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import database as db  # Import your new database manager

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Mushroom Yield Optimizer",
    page_icon="🍄",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- INITIALIZE DATABASE ---
# Creates the file 'farm_data.db' if it doesn't exist
db.init_db()

# --- CACHED ASSET LOADING ---
@st.cache_resource
def load_model():
    # Construct path relative to this file
    model_path = os.path.join(os.path.dirname(__file__), "artifacts", "mushroom_brain_v2.keras")
    
    if not os.path.exists(model_path):
        st.error(f"❌ File not found: {model_path}")
        return None
        
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_tools():
    base_path = os.path.join(os.path.dirname(__file__), "artifacts")
    prep_path = os.path.join(base_path, "preprocessor_v2.pkl")
    label_path = os.path.join(base_path, "labels_v2.pkl")
    
    if not os.path.exists(prep_path):
        st.error(f"❌ File not found: {prep_path}")
        return None, None
        
    preprocessor = joblib.load(prep_path)
    labels = joblib.load(label_path)
    return preprocessor, labels

# Load assets immediately
try:
    model = load_model()
    preprocessor, labels_data = load_tools()
    
    if model is None or preprocessor is None:
        st.stop()
    
    # Extract column definitions
    CAT_COLS = labels_data['cat_columns']
    NUM_COLS = labels_data['num_columns']
    
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# --- SANITIZATION ENGINE ---
def sanitize_data(df, num_cols, cat_cols):
    df_clean = df.copy()
    # 1. Force Numerical
    for col in num_cols:
        if col not in df_clean.columns:
            df_clean[col] = 0
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    # 2. Force Categorical
    for col in cat_cols:
        if col not in df_clean.columns:
            df_clean[col] = "Unknown"
        df_clean[col] = df_clean[col].astype(str).fillna("Unknown")
    return df_clean

# --- UI HEADER ---
st.title("🍄 Mushroom Manager Pro")
st.markdown("Manage batches and predict yields.")

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["➕ New Batch & Analyze", "📊 Farm Dashboard"])

# ==========================
# TAB 1: INPUT & PREDICTION
# ==========================
with tab1:
    with st.form("prediction_form"):
        st.info("ℹ️ Enter batch details below to get AI predictions and save to history.")
        
        # New Field: Batch Name for tracking
        batch_name = st.text_input("Batch Name / ID", value="Batch-001", help="Give this batch a unique name")

        with st.expander("Stage 1: Bagging & Substrate", expanded=True):
            c1, c2 = st.columns(2)
            # UPDATE THESE TO MATCH YOUR TRAINING DATA COLUMNS
            bag_temp = c1.number_input("Substrate Temp (°C)", min_value=0.0, value=25.0)
            substrate_type = c2.selectbox("Substrate Type", ["Oak", "Straw", "Soy", "Mix"])
        
        with st.expander("Stage 2: Inoculation"):
            c1, c2 = st.columns(2)
            spawn_rate = c1.number_input("Spawn Rate (%)", min_value=0.0, value=5.0)
            strain_id = c2.text_input("Strain ID", value="M-2024-X")

        with st.expander("Stage 3: Growth Cycle"):
            c1, c2 = st.columns(2)
            humidity_avg = c1.slider("Avg Humidity (%)", 0, 100, 85)
            co2_level = c2.number_input("CO2 Level (ppm)", value=800)

        with st.expander("Stage 4: Harvest Conditions"):
            c1, c2 = st.columns(2)
            days_to_pin = c1.number_input("Days to Pinning", value=14)
            cap_color = c2.selectbox("Cap Color Check", ["Normal", "Pale", "Dark"])

        submitted = st.form_submit_button("Analyze & Save Batch", type="primary")

    if submitted:
        # 1. Construct DataFrame for Model
        input_data = {
            'substrate_temp': bag_temp,
            'spawn_rate': spawn_rate,
            'humidity': humidity_avg,
            'co2_level': co2_level,
            'days_to_pin': days_to_pin,
            'substrate_type': substrate_type,
            'strain_id': strain_id,
            'cap_color': cap_color,
        }
        
        raw_df = pd.DataFrame([input_data])
        clean_df = sanitize_data(raw_df, NUM_COLS, CAT_COLS)
        
        try:
            # 2. Run Prediction
            processed_data = preprocessor.transform(clean_df)
            predictions = model.predict(processed_data)
            
            pred_yield = float(predictions[0][0])
            pred_issue_prob = float(predictions[1][0])
            
            # 3. Save to Database
            db_entry = {
                "batch_name": batch_name,
                "substrate_type": substrate_type,
                "strain_id": strain_id,
                "pred_yield": pred_yield,
                "pred_prob": pred_issue_prob
            }
            db.add_batch(db_entry)
            
            # 4. Show Results
            st.divider()
            st.success(f"✅ Batch '{batch_name}' Saved Successfully!")
            
            st.subheader("AI Analysis Results")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Yield", f"{pred_yield:.2f} kg")
            
            if pred_issue_prob > 0.5:
                col2.error(f"⚠️ Risk Detected: {pred_issue_prob:.1%}")
            else:
                col2.success(f"✅ Health Score: {1 - pred_issue_prob:.1%}")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ==========================
# TAB 2: DASHBOARD
# ==========================
with tab2:
    st.header("Farm History")
    
    # Load data from SQLite
    df_history = db.get_all_batches()
    
    if not df_history.empty:
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Batches", len(df_history))
        m2.metric("Avg Predicted Yield", f"{df_history['predicted_yield'].mean():.2f} kg")
        m3.metric("Latest Batch", df_history['batch_name'].iloc[0])
        
        st.divider()
        
        # Data Table
        st.dataframe(
            df_history, 
            use_container_width=True,
            column_config={
                "predicted_yield": st.column_config.NumberColumn("Yield (kg)", format="%.2f"),
                "issue_prob": st.column_config.ProgressColumn("Risk Probability", format="%.2f", min_value=0, max_value=1),
                "timestamp": "Date Recorded"
            }
        )
        
        # Download Button
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Data (CSV)", 
            data=csv, 
            file_name="my_farm_data.csv", 
            mime="text/csv"
        )
    else:
        st.info("📭 No data found. Go to the 'New Batch' tab to add your first entry!")