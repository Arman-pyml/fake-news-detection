import streamlit as st
import pickle
from models.train_model import clean_text   # agar model.py me hai
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<h1 style='text-align: center; color: #22c55e;'>
📰 Fake News Detector
</h1>
<h4 style='text-align: center; color: #94a3b8;'>
Real-time Fake News Detection using Machine Learning & NLP
</h4>
<hr>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## 📰 Fake News Detector")
    st.markdown("### ML + NLP Project")
    st.markdown("---")
    st.info("Real-time ML based news verification")

    option = st.radio(
        "Choose Input Method",
        ("Paste News Text", "Upload News File")
    )

st.title("🧠 Fake News Detection System")
st.caption("Machine Learning based | Real-Time Ready")
st.markdown("---")

if option == "Paste News Text":
    news_text = st.text_area(
        "📝 Paste News Here",
        height=200,
        placeholder="Enter news content to verify..."
    )


    check_btn = st.button("🔍 Check News")

    st.markdown("### 📊 Prediction Result")

    
with st.expander("ℹ️ Model Information"):
    st.write("""
    - Algorithm: Logistic Regression / Naive Bayes
    - Dataset: Kaggle Fake News Dataset
    - Features: TF-IDF
    - Accuracy: ~95%
    """)
st.markdown("---")
st.caption("Made by Arman Ansari | ML Project")


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    }
    </style>
    """,
    unsafe_allow_html=True
)

import streamlit as st

# ===== Dark / Light Mode Toggle =====
theme = st.sidebar.radio(
    "🌗 Select Theme",
    ("Light Mode", "Dark Mode")
)

if theme == "Dark Mode":
    st.markdown("""
        <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .stApp {
            background-color: #0e1117;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {
            background-color: white;
            color: black;
        }
        .stApp {
            background-color: white;
        }
        </style>
    """, unsafe_allow_html=True)
st.sidebar.title("📰 Fake News Detector")
st.sidebar.info("Real-time + ML based news verification")

st.title("📰 Fake News Detection System")
st.subheader("ML + Real-Time News Verification")
st.write("Paste news text or fetch live news to check authenticity.")


st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background-color: #020617;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #22c55e;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<h1 style='text-align: center; color: #22c55e;'>
📰 Fake News Detection System
</h1>
<h4 style='text-align: center; color: #94a3b8;'>
AI-powered Real-Time News Verification
</h4>
<hr>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.result-card {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
}
.real {
    background-color: #14532d;
    color: #bbf7d0;
}
.fake {
    background-color: #7f1d1d;
    color: #fecaca;
}
.confidence {
    font-size: 16px;
    margin-top: 10px;
    color: #e5e7eb;
}
</style>
""", unsafe_allow_html=True)



# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection App")

news_text = st.text_area(
    "News yahan likho 👇",
    height=200,
    placeholder="Example: Virat Kohli has retired from Test cricket..."
)

check_btn = st.button("🔍 Predict")

if check_btn:

    if news_text.strip() == "":
        st.warning("⚠️ Please enter news text")

    else:
        cleaned = clean_text(news_text)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)
        probability = model.predict_proba(vector)
        confidence = max(probability[0]) * 100

        if confidence < 60:
            st.warning(
                f"⚠️ Result Uncertain ({confidence:.2f}%)\n"
                "This may be real-time or recent news.\n"
                "Please verify from trusted sources."
            )

        elif prediction[0] == 1:
            st.success(f"🟢 REAL NEWS\nConfidence: {confidence:.2f}%")

        else:
            st.error(f"🔴 FAKE NEWS\nConfidence: {confidence:.2f}%")

