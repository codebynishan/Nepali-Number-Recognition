import streamlit as st
import requests

st.set_page_config(page_title="Nepali Digit Recognition", page_icon="🖼️")

st.title("Nepali Digit Recognition")

st.markdown("Upload an image of a Nepali Number .")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

api_url = "http://127.0.0.1:8000/predict"
api_key = "MY_API_KEY"  

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
               
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

if st.button("Predict using Agent"):
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    headers = {"x-api-key": api_key}

    with st.spinner("Predicting using agent..."):
        response = requests.post(
            "http://127.0.0.1:8000/predict?use_agent=true",
            files=files,
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()["class"]
            if result == "unknown class":
                st.warning("⚠️ This image does not belong to known Nepali digits.")
            else:
                st.success(f"Agent Prediction: {result}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

