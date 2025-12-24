import streamlit as st
import requests

st.set_page_config(page_title="Nepali Digit Recognition", page_icon="🖼️")

st.title("Nepali Digit Recognition")

st.markdown("Upload an image of a Nepali Number .")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

api_url = "http://127.0.0.1:8000/predict"
api_key = "mysecretapikey"  

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict using Model"):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        headers = {"x-api-key": api_key}
        with st.spinner("Predicting..."):
            response = requests.post(api_url, files=files, headers=headers)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['class']}")
                # st.info(f"Confidence: {result['confidence']:.2f}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

   
    if st.button("Predict using Agent"):
        st.warning("Agent-based prediction not implemented yet. Use model prediction.")
