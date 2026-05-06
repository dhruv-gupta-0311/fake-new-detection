import streamlit as st
import joblib 
import numpy as np
from src.data_processor import DataProcessor
from src.bert_predictor import BertPredictor
st.set_page_config(page_title="Fake News Detector")
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/logistic_model.joblib')
        vectorizer = joblib.load('models/tfidf_vectorized.joblib')
        processor = DataProcessor()
        bert_model = BertPredictor('models/distilbert_finetuned')
        return model, vectorizer, processor, bert_model
    except FileNotFoundError as e:
        st.error(f"Missing asset: {e}")
        st.info("Run 'uv run main.py'")
        st.stop()
    except AttributeError as e:
        st.error(f"Version mismatch: {e}")
        st.info("Delete models/ and re-run `uv run main.py`.")
        st.stop()
m, v, p, b = load_assets()
def get_explaination(cleaned_text, model, vectorizer, top_n=8):
    vec = vectorizer.transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    non_zero_indices = vec.nonzero()[1]
    word_contributions = []
    for idx in non_zero_indices:
        word = feature_names[idx]
        tfidf_value = vec[0, idx]
        coefficient = coefficients[idx]
        contribution = float(tfidf_value * coefficient)
        word_contributions.append((word, contribution))
    word_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    # Negative = pushes toward Real (label=0)
    # Positive = pushes toward Fake (label=1)
    real_pushers = [(w, abs(s)) for w, s in word_contributions if s < 0][:top_n]
    fake_pushers = [(w, abs(s)) for w, s in word_contributions if s > 0][:top_n]
    return real_pushers, fake_pushers
def predict(text, mode):
    cleaned = p.clean_text(text)
    vec = v.transform([cleaned])
    lr_pred = m.predict(vec)[0]
    lr_prob = m.predict_proba(vec)[0]
    lr_conf = float(lr_prob[lr_pred])
    if mode == "TF-IDF + Logistic Regression":
        real_pushers, fake_pushers = get_explaination(cleaned, m, v)
        return lr_pred, lr_conf, lr_conf, "TF-IDF", real_pushers, fake_pushers
    elif mode == "DistilBERT":
        bert_pred, bert_proba, bert_conf = b.predict(text)
        return bert_pred, bert_conf, bert_proba, "DistilBERT", [], []
    else:
        if lr_conf >= 0.85:
            real_pushers, fake_pushers = get_explaination(cleaned, m, v)
            return lr_pred, lr_prob, lr_conf, "TF-IDF (High Confidence)", real_pushers, fake_pushers
        bert_pred, bert_proba, bert_conf = b.predict(text)
        LR_W, BERT_W = 0.25, 0.75
        combined = np.array([LR_W * lr_prob[i] + BERT_W * bert_proba[i] for i in range(2)])
        final_pred = int(np.argmax(combined))
        final_conf = float(combined[final_pred].item())
        return final_pred, final_conf, combined, "Hybrid (LR + BERT)", [], []
for key, default in {
    "prediction": None,
    "confidence": None,
    "real_pushers": [],
    "fake_pushers": [],
    "model_used": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
with st.sidebar:
    st.title("Model Info")

    st.metric("Dataset Size", "72,000 rows")
    st.metric("Model", "TF-IDF + Logistic Regression")

    st.divider()

    # FIXED: Honest accuracy presentation
    st.markdown("**Performance**")
    st.metric("In-Distribution Accuracy", "94%")
    st.caption(
        "Measured on US political news (2015–2018). "
        "Accuracy on other domains will vary."
    )

    st.divider()

    st.markdown("**Known Limitations**")
    st.caption("Reliable on: US political news, obvious clickbait.")
    st.caption("Unreliable on: Science, finance, non-US sources, short headlines.")
    st.caption("Cannot detect: Sophisticated disinformation mimicking journalistic format.")

    st.divider()
    st.caption(
        "This model detects journalistic format patterns, "
        "not factual truth. A well-formatted false story "
        "may score as Real News."
    )
    st.divider()
    mode = st.radio("Prediction Mode", ["TF-IDF + Logistic Regression", "DistilBERT", "Hybrid (Auto)"], index=0)
    
st.title("Fake News Detection")
user_input = st.text_area("Enter the article text here: ", height=250, placeholder="Avoid pasting short headlines or single sentences. The model performs best on longer text with clear journalistic formatting.")
col1, col2 = st.columns([1, 4])
with col1:
    predict_btn = st.button("Predict", type="primary", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        
        for key in ["prediction", "confidence", "real_pushers", "fake_pushers"]:
            st.session_state[key] = None
        st.rerun()
if predict_btn:
    if user_input:
        with st.spinner("Predicting..."):
            # user_input_cleaned = p.clean_text(user_input)
            # user_input_vectorized = v.transform([user_input_cleaned])
            # prediction = m.predict(user_input_vectorized)[0]
            # probability = m.predict_proba(user_input_vectorized)[0]
            # confidence = float(probability[prediction])
            # real_pushers, fake_pushers = get_explaination(user_input_cleaned, m, v)
            # st.session_state.prediction = int(prediction)
            # st.session_state.confidence = confidence
            # st.session_state.real_pushers = real_pushers
            # st.session_state.fake_pushers = fake_pushers
            pred, conf, proba, model_used, real_pushers, fake_pushers = predict(user_input, mode)
            st.session_state.prediction = int(pred)
            st.session_state.confidence = conf
            st.session_state.real_pushers = real_pushers
            st.session_state.fake_pushers = fake_pushers
            st.session_state.model_used = model_used
            # if st.session_state.prediction is not None:
            #     st.divider()
            #     pred = st.session_state.prediction
            #     conf = st.session_state.confidence
            # if conf < 0.6:
            #     st.warning(f"Low confidence: {conf:.1%}")
            # elif pred == 0:
            #     st.success(" Prediction: Real News")
            #     st.metric(label="Confidence", value=f"{probability[0]*100:.2f}%")
            # else:
            #     st.error(" Prediction: Fake News")
            #     st.metric(label="Confidence", value=f"{probability[1]*100:.2f}%")
    else:
        st.warning("Enter text to predict.")
    if st.session_state.prediction is not None:
        st.divider()
        pred = st.session_state.prediction
        conf = st.session_state.confidence
        if pred == 0:
            st.success(" Prediction: Real News")
        else:
            st.error(" Prediction: Fake News")
        if conf < 0.65:
            st.warning(f"Low confidence: {conf:.1%}")
        st.metric(label="Confidence", value=f"{conf:.2%}")
        st.progress(conf)
        st.caption(f"Model used: {st.session_state.model_used}")
    if st.session_state.real_pushers or st.session_state.fake_pushers:
        with st.expander("Detailed Decision Analysis", expanded=True):
            st.caption(
                "Shows which words in your text most influenced the prediction. "
                "Impact score = TF-IDF weight × learned coefficient."
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("** Pushing toward Real:**")
                if st.session_state.real_pushers:
                    for word, score in st.session_state.real_pushers:
                        bar = min(int(score * 500), 15)
                        st.markdown(f"`{word}` {bar} `{score:.4f}`")
                else:
                    st.caption("No significant Real signals found.")

            with col2:
                st.markdown("** Pushing toward Fake:**")
                if st.session_state.fake_pushers:
                    for word, score in st.session_state.fake_pushers:
                        bar = min(int(score * 500), 15)
                        st.markdown(f"`{word}` {bar} `{score:.4f}`")
                else:
                    st.caption("No significant Fake signals found.")

            st.divider()
    else:
        st.info("Prediction by Distilbert does not provide word-level explanations.")
        
    
                

    