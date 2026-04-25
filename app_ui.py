import streamlit as st
import joblib 
from src.data_processor import DataProcessor
st.set_page_config(page_title="Fake News Detector")
@st.cache_resource
def load_assets():
    model = joblib.load('models/logistic_model.joblib')
    vectorizer = joblib.load('models/tfidf_vectorized.joblib')
    processor = DataProcessor()
    return model, vectorizer, processor
m, v, p = load_assets()
with st.sidebar:
    st.title("Stats")
    st.metric("Dataset size", "72k")
    st.metric("Test Accuracy", "93%")
    
st.title("Fake News Detection")
user_input = st.text_area("Enter the article text here: ", height=250)

if st.button("Predict"):
    if user_input:
        with st.spinner("Predicting..."):
            #user_input_cleaned = p.clean_text(user_input)
            user_input_cleaned = user_input
            user_input_vectorized = v.transform([user_input_cleaned])
            prediction = m.predict(user_input_vectorized)[0]
            probability = m.predict_proba(user_input_vectorized)[0]
            if prediction == 0:
                st.error(f"Prediction: Fake News")
                st.metric(label="Confidence", value=f"{probability[0]*100:.2f}%")
            else:
                st.success("Prediction: Real News")
                st.metric(label="Confidence", value=f"{probability[1]*100:.2f}%")
    else:
        st.warning("Enter text to predict.")
                

    