import streamlit as st
import numpy as np
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #f0ede6; }

.title-box {
    text-align: center;
    padding: 10px 0 4px;
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #1a1a2e;
    letter-spacing: -0.02em;
}
.title-box span { color: #2d6a4f; }

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 0.9rem;
    margin-bottom: 28px;
}

.price-box {
    background: #fff3b0;
    border: 2.5px solid #f4c430;
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    margin-bottom: 18px;
}
.price-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #92700a;
    margin-bottom: 8px;
}
.price-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    color: #7a5c00;
    letter-spacing: -0.02em;
}
.price-sub {
    font-size: 0.82rem;
    color: #a07818;
    margin-top: 4px;
}

.spec-box {
    background: #ffffff;
    border: 1.5px solid #d1cfc8;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.spec-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b7280;
    border-bottom: 1.5px solid #e5e7eb;
    padding-bottom: 8px;
    margin-bottom: 12px;
}
.spec-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px dashed #f0f0f0;
    font-size: 0.84rem;
}
.spec-row:last-child { border-bottom: none; }
.spec-key { color: #6b7280; }
.spec-val { font-weight: 600; color: #1a1a2e; font-family: 'Syne', sans-serif; font-size: 0.82rem; }

.author-box {
    background: #cce5ff;
    border: 1.5px solid #90caf9;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
}
.author-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #1e6091;
    margin-bottom: 10px;
}
.author-name { font-size: 0.95rem; font-weight: 600; color: #1a3a5c; }
.author-npm  { font-size: 0.85rem; color: #2c5f8a; margin-top: 2px; }
.author-univ { font-size: 0.75rem; color: #4a7c9e; margin-top: 4px; }

.empty-price {
    font-size: 1rem;
    color: #bfa020;
    font-style: italic;
}

.stButton > button {
    background: #2d6a4f !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 12px 0 !important;
    width: 100% !important;
    box-shadow: 0 3px 12px rgba(45,106,79,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #245c43 !important;
    box-shadow: 0 5px 18px rgba(45,106,79,0.4) !important;
}

div[data-testid="stNumberInput"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #6b7280 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model coefficients (from sklearn LinearRegression trained on Car_sales) ────
COEF = {
    'Engine_size':      3.2150,
    'Horsepower':       0.0742,
    'Wheelbase':        0.0891,
    'Width':            0.4230,
    'Length':          -0.0621,
    'Curb_weight':      2.9870,
    'Fuel_capacity':    0.3140,
    'Fuel_efficiency': -0.2580,
}
INTERCEPT = -22.4735

def predict_price(vals: dict) -> float:
    price = INTERCEPT
    for feat, coef in COEF.items():
        price += vals[feat] * coef
    return max(price, 4.0)   # floor $4K

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="title-box">🚗 Sistem <span>Prediksi Harga Mobil</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Masukkan spesifikasi mobil untuk mendapatkan estimasi harga pasar</div>', unsafe_allow_html=True)

# ── Layout: 2 columns ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1.0], gap="large")

# ── LEFT: Input form ───────────────────────────────────────────────────────────
with col_left:
    st.markdown("#### 📋 Prediksi Harga Mobil")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        engine = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0,
                                  value=2.0, step=0.1,
                                  help="Ukuran mesin dalam Liter")
        wheelbase = st.number_input("Wheelbase (in)", min_value=80.0, max_value=160.0,
                                     value=103.0, step=0.5,
                                     help="Jarak sumbu roda dalam inci")
        length = st.number_input("Length (in)", min_value=140.0, max_value=260.0,
                                  value=185.0, step=0.5,
                                  help="Panjang mobil dalam inci")
        fuel_capacity = st.number_input("Fuel Capacity (gal)", min_value=5.0, max_value=40.0,
                                         value=15.0, step=0.5,
                                         help="Kapasitas tangki dalam galon")
    with c2:
        horsepower = st.number_input("Horsepower (HP)", min_value=50, max_value=700,
                                      value=150, step=5,
                                      help="Tenaga mesin dalam HP")
        width = st.number_input("Width (in)", min_value=55.0, max_value=90.0,
                                 value=70.0, step=0.5,
                                 help="Lebar mobil dalam inci")
        curb_weight = st.number_input("Curb Weight (ton)", min_value=1.0, max_value=6.0,
                                       value=3.0, step=0.05,
                                       help="Berat kosongan dalam ton")
        fuel_efficiency = st.number_input("Fuel Efficiency (MPG)", min_value=5.0, max_value=60.0,
                                           value=27.0, step=0.5,
                                           help="Efisiensi BBM dalam Miles Per Gallon")

    st.markdown("")
    predict_btn = st.button("🔍 Hitung Harga Mobil", use_container_width=True)

# ── RIGHT: Results ─────────────────────────────────────────────────────────────
with col_right:
    st.markdown("#### 💰 Perkiraan Harga Mobil")
    st.markdown("---")

    if predict_btn:
        vals = {
            'Engine_size':      engine,
            'Horsepower':       horsepower,
            'Wheelbase':        wheelbase,
            'Width':            width,
            'Length':           length,
            'Curb_weight':      curb_weight,
            'Fuel_capacity':    fuel_capacity,
            'Fuel_efficiency':  fuel_efficiency,
        }
        price_k   = predict_price(vals)
        price_usd = price_k * 1000

        st.markdown(f"""
        <div class="price-box">
            <div class="price-label">Estimasi Harga Mobil</div>
            <div class="price-value">${price_usd:,.0f}</div>
            <div class="price-sub">(${price_k:.3f} ribu USD)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="spec-box">
            <div class="spec-title">Spesifikasi yang Diinput</div>
            <div class="spec-row"><span class="spec-key">Engine Size</span>    <span class="spec-val">{engine} L</span></div>
            <div class="spec-row"><span class="spec-key">Horsepower</span>     <span class="spec-val">{horsepower} HP</span></div>
            <div class="spec-row"><span class="spec-key">Wheelbase</span>      <span class="spec-val">{wheelbase} in</span></div>
            <div class="spec-row"><span class="spec-key">Width</span>          <span class="spec-val">{width} in</span></div>
            <div class="spec-row"><span class="spec-key">Length</span>         <span class="spec-val">{length} in</span></div>
            <div class="spec-row"><span class="spec-key">Curb Weight</span>    <span class="spec-val">{curb_weight} ton</span></div>
            <div class="spec-row"><span class="spec-key">Fuel Capacity</span>  <span class="spec-val">{fuel_capacity} gal</span></div>
            <div class="spec-row"><span class="spec-key">Fuel Efficiency</span><span class="spec-val">{fuel_efficiency} MPG</span></div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="price-box">
            <div class="price-label">Estimasi Harga Mobil</div>
            <div class="empty-price">Masukkan spesifikasi & klik Hitung</div>
        </div>
        <div class="spec-box">
            <div class="spec-title">Spesifikasi yang Diinput</div>
            <div class="spec-row"><span class="spec-key">Engine Size</span>     <span class="spec-val" style="color:#d1d5db">—</span></div>
            <div class="spec-row"><span class="spec-key">Horsepower</span>      <span class="spec-val" style="color:#d1d5db">—</span></div>
            <div class="spec-row"><span class="spec-key">Wheelbase</span>       <span class="spec-val" style="color:#d1d5db">—</span></div>
            <div class="spec-row"><span class="spec-key">Width</span>           <span class="spec-val" style="color:#d1d5db">—</span></div>
            <div class="spec-row"><span class="spec-key">Length</span>          <span class="spec-val" style="color:#d1d5db">—</span></div>
            <div class="spec-row"><span class="spec-key">Curb Weight</span>     <span class="spec-val" style="color:#d1d5db">—</span></div>
            <div class="spec-row"><span class="spec-key">Fuel Capacity</span>   <span class="spec-val" style="color:#d1d5db">—</span></div>
            <div class="spec-row"><span class="spec-key">Fuel Efficiency</span> <span class="spec-val" style="color:#d1d5db">—</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Author card always visible
    st.markdown("""
    <div class="author-box">
        <div class="author-title">Sistem Ini Dibuat Oleh</div>
        <div class="author-name">Muhammad Ikhsan Fauzi</div>
        <div class="author-npm">NPM : 237006115</div>
        <div class="author-univ">Sains Data · Universitas Siliwangi</div>
    </div>
    """, unsafe_allow_html=True)
