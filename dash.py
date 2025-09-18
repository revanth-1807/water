import streamlit as st
import numpy as np
import pandas as pd
import pickle
from geopy.geocoders import Nominatim
import pydeck as pdk
from streamlit_js_eval import get_geolocation
import geocoder
import time # 👈 Import the time library

# -----------------------------
# Custom Card Components (No changes here)
# -----------------------------
def safe_card(quality, disease):
    st.markdown(f"""
    <div style="
        padding:20px;
        border-radius:15px;
        background: linear-gradient(135deg, #E3FCEC, #C8F7C5);
        border-left: 8px solid #2E7D32;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;">
        <h2 style="color:#2E7D32; margin:0;">✅ Water is Safe</h2>
        <p style="font-size:18px; margin:5px 0;">
            <b>Quality:</b> {quality}<br>
            <b>Disease Risk:</b> {disease}
        </p>
    </div>
    """, unsafe_allow_html=True)


def unsafe_card(quality, disease, area):
    st.markdown(f"""
    <div style="
        padding:20px;
        border-radius:15px;
        background: linear-gradient(135deg, #FFE3E3, #FFBABA);
        border-left: 8px solid #C62828;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;">
        <h2 style="color:#C62828; margin:0;">❌ Unsafe Water</h2>
        <p style="font-size:18px; margin:5px 0;">
            <b>Quality:</b> {quality}<br>
            <b>Disease Risk:</b> {disease}<br>
            <b>Alert:</b> Residents of {area} are notified 🚨
        </p>
    </div>
    """, unsafe_allow_html=True)


def warning_card(msg):
    st.markdown(f"""
    <div style="
        padding:20px;
        border-radius:15px;
        background: linear-gradient(135deg, #FFF9E3, #FFE7A0);
        border-left: 8px solid #FF9800;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;">
        <h2 style="color:#E65100; margin:0;">⚠ Warning</h2>
        <p style="font-size:18px; margin:5px 0;">{msg}</p>
    </div>
    """, unsafe_allow_html=True)

# Try importing TensorFlow safely
try:
    import tensorflow as tf
    tf_version = tf.__version__  # ✅ fixed
    st.sidebar.info(f"✅ TensorFlow {tf_version} loaded successfully")
except ImportError:
    st.sidebar.error("❌ TensorFlow not installed or version mismatch")
    tf = None

# -----------------------------
# Load model and preprocessing tools safely (No changes here)
# -----------------------------
model, scaler, le_quality, le_diseases = None, None, None, None
try:
    if tf:
        model = tf.keras.models.load_model("best_model.keras")
        scaler = pickle.load(open("scaler.pkl", "rb"))
        le_quality = pickle.load(open("le_quality.pkl", "rb"))
        le_diseases = pickle.load(open("le_diseases.pkl", "rb"))
except Exception as e:
    st.sidebar.warning(f"⚠ Model not loaded: {e}")
    model = None

# -----------------------------
# Streamlit UI Setup (No changes here, except for the new section)
# -----------------------------
st.set_page_config(page_title="Water Quality & Disease Prediction", page_icon="💧", layout="wide")
st.sidebar.title("⚙ Settings")
st.sidebar.markdown("Use the sidebar to customize parameters.")
st.title("💧 Water Quality & Disease Prediction Dashboard")
st.markdown("This dashboard predicts water safety and potential diseases based on sensor readings.")

col1,spacer, col2 = st.columns([1,0.2, 2])

with col1:
    st.subheader("📍 Location Info")

    location = get_geolocation()

    if location:
        lat = location["coords"]["latitude"]
        lon = location["coords"]["longitude"]
        st.success(f"✅ Auto-detected Location: {lat}, {lon}")
    else:
        g = geocoder.ip('me')
        if g.ok:
            lat, lon = g.latlng
            st.info(f"🌍 Approx Location from IP: {lat}, {lon}")
        else:
            st.warning("⚠ Could not detect location. Please enter manually.")
            lat = st.number_input("Latitude", value=16.836565, format="%.6f")
            lon = st.number_input("Longitude", value=81.517963, format="%.6f")

    geolocator = Nominatim(user_agent="water_app")
    try:
        loc = geolocator.reverse((lat, lon), language='en', timeout=10)
        area = loc.address if loc else "Unknown area"
    except:
        area = "Unknown area"

    st.info(f"📌 Detected Area: {area}")


    st.subheader("💦 Water Parameters")
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
    hardness = st.number_input("Hardness", min_value=0.0)
    solids = st.number_input("Solids", min_value=0.0)
    chloramines = st.number_input("Chloramines", min_value=0.0)
    sulfate = st.number_input("Sulfate", min_value=0.0)
    conductivity = st.number_input("Conductivity", min_value=0.0)
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0)
    trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0)
    turbidity = st.number_input("Turbidity", min_value=0.0)

    predict_btn = st.button("🔮 Predict", use_container_width=True)

with col2:
    st.subheader("📈 Water Impurity Over Time")
    time_steps = np.arange(1, 11)
    impurity_rate = np.cumsum(np.random.randint(1, 5, size=10))
    df_chart = pd.DataFrame({
        "Time": time_steps,
        "Impurity Rate": impurity_rate
    })
    st.line_chart(df_chart.set_index("Time"))

    # -----------------------------
    # NEW: Flowchart with glowing effect
    # -----------------------------
    st.markdown("""
        <style>
            @keyframes glow {
                0% { box-shadow: 0 0 5px rgba(0,255,0,0.7), 0 0 10px rgba(0,255,0,0.5); }
                50% { box-shadow: 0 0 15px rgba(0,255,0,1), 0 0 20px rgba(0,255,0,0.8); }
                100% { box-shadow: 0 0 5px rgba(0,255,0,0.7), 0 0 10px rgba(0,255,0,0.5); }
            }
            .flow-box {
                border: 2px solid #555;
                border-radius: 10px;
                padding: 10px 20px;
                margin: 10px auto;
                width: 150px;
                text-align: center;
                font-weight: bold;
                font-size: 1.2em;
                transition: all 0.3s ease-in-out;
            }
            .flow-arrow {
                height: 30px;
                display: block;
                margin: 0 auto;
            }
            .glowing-box {
                animation: glow 1.5s infinite ease-in-out;
            }
            .glowing-arrow {
                animation: glow 1.5s infinite ease-in-out;
            }
            .arrow-path {
                fill: none;
                stroke: #555;
                stroke-width: 2px;
                transition: stroke 0.3s ease-in-out;
            }
            .glowing-arrow .arrow-path {
                stroke: #0F0; /* Green glow color for the arrow */
            }
        </style>
    """, unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown(
    "<h2 style='text-align: center;'>🔄 Prediction Flow</h2>",
    unsafe_allow_html=True
)
    
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    st.write("")
    st.write("")
    st.write("")
    st.write("")
 
    

    
    # Placeholders for dynamic updates
    input_box = st.empty()
    arrow1_svg = st.empty()
    predict_box = st.empty()
    arrow2_svg = st.empty()
    output_box = st.empty()
    
    # Function to draw SVG arrow
    def draw_arrow(glowing=False):
        stroke_color = '#0F0' if glowing else '#555'
        return f"""
            <svg width="100%" height="30" viewBox="0 0 100 30">
              <path class="arrow-path" d="M50,0 V25 M50,25 L45,20 M50,25 L55,20" style="stroke:{stroke_color};" />
            </svg>
        """

    # Initial state: Input box is glowing
    input_box.markdown(f'<div class="flow-box glowing-box">Input</div>', unsafe_allow_html=True)
    arrow1_svg.markdown(draw_arrow(), unsafe_allow_html=True)
    predict_box.markdown(f'<div class="flow-box">Predict</div>', unsafe_allow_html=True)
    arrow2_svg.markdown(draw_arrow(), unsafe_allow_html=True)
    output_box.markdown(f'<div class="flow-box">Output</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction Logic
# -----------------------------
if predict_btn:
    st.subheader("📊 Prediction Results")

    # Change state to "Predict" (Input stops glowing, Predict starts)
    with col2:
        input_box.markdown(f'<div class="flow-box">Input</div>', unsafe_allow_html=True)
        arrow1_svg.markdown(draw_arrow(glowing=True), unsafe_allow_html=True)
        predict_box.markdown(f'<div class="flow-box glowing-box">Predict</div>', unsafe_allow_html=True)
        output_box.markdown(f'<div class="flow-box">Output</div>', unsafe_allow_html=True)
        # Simulate processing time
        time.sleep(1)

    if model and scaler:
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                                conductivity, organic_carbon, trihalomethanes, turbidity]])
        input_scaled = scaler.transform(input_data)

        yq_pred, yd_pred = model.predict(input_scaled)
        quality_label = le_quality.inverse_transform([np.argmax(yq_pred)])
        disease_label = le_diseases.inverse_transform([np.argmax(yd_pred)])

        quality_class = quality_label[0]
        disease_class = disease_label[0]
    else:
        st.warning("⚠ Using default logic as model not available.")
        quality_class = "Safe" if 6.5 <= ph <= 8.5 else "Bad"
        disease_class = "None" if quality_class == "Safe" else "Cholera"

    # Change state to "Output" (Predict stops glowing, Output starts)
    with col2:
        input_box.markdown(f'<div class="flow-box">Input</div>', unsafe_allow_html=True)
        arrow1_svg.markdown(draw_arrow(), unsafe_allow_html=True)
        predict_box.markdown(f'<div class="flow-box">Predict</div>', unsafe_allow_html=True)
        arrow2_svg.markdown(draw_arrow(glowing=True), unsafe_allow_html=True)
        output_box.markdown(f'<div class="flow-box glowing-box">Output</div>', unsafe_allow_html=True)
        time.sleep(1)

    # Set map color based on prediction
    if quality_class.lower() == "bad":
        st.error(f"❌ Predicted Quality: {quality_class}")
        st.warning(f"⚠ Predicted Disease Risk: {disease_class}")
        st.info(f"📢 Broadcast Alert: Residents of {area} are notified (simulated).")
        color = [255, 0, 0]  # Red for unsafe
    else:
        st.success(f"✅ Predicted Quality: {quality_class}")
        st.success(f"🩺 Predicted Disease Risk: {disease_class}")
        color = [0, 0, 255]  # Blue for safe

    # Update map with prediction color
    df_map = pd.DataFrame({
        "lat": [lat],
        "lon": [lon],
        "color_r": [color[0]],
        "color_g": [color[1]],
        "color_b": [color[2]],
    })

    st.pydeck_chart(
        pdk.Deck(
            map_style="light",
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=12),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_map,
                    get_position="[lon, lat]",
                    get_color="[color_r, color_g, color_b]",
                    get_radius=600,
                )
            ],
        )
    )

    st.caption("🔵 Safe water  |  🔴 Unsafe water")