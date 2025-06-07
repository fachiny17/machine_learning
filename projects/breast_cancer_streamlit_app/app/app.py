import streamlit as st
import pandas as pd
import numpy as np
from joblib import dump, load
import plotly.graph_objects as go

from typing import Generator
from groq import Groq
import json
import os
import uuid

# Initialize Groq client
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
    
# Model configuration
MODEL_CONFIG = {
    "name": "Compound-Beta",
    "tokens": 8192,
    "developer": "Groq"
}

# File to store chat history
HISTORY_FILE = "chat_history.json"

def get_clean_data():
    data = pd.read_csv("../data/data.csv")    
    data = data.drop(['Unnamed: 32', 'id'], axis=1)    
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
        ]
    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict  

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter',
                  'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
          theta=categories,
          fill='toself',
          name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
            input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
            input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
            input_data['fractal_dimension_se']
        ],
          theta=categories,
          fill='toself',
          name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
          theta=categories,
          fill='toself',
          name='Worst Value'
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
      showlegend=True
    )

    return fig

def add_predictions(input_data):
    model = load('../models/logistic_regression_model.joblib')
    scaler = load('../models/scaler.joblib')

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is: ")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])

# Section 1: NucleiScan AI Predictor
def nuclei_predict():
    input_data = add_sidebar()

    with st.container():
        st.title("NucleiScan AI: Breast Cancer Predictor")
        st.write("NucleiScan AI analyzes 30 cell nuclei characteristics from breast tissue samples to predict malignancy risk. This clinical decision-support tool evaluates mean, standard error, and worst-case measurements (including radius, concavity, and texture) to assist pathologists in rapid assessment. Features interactive sliders for manual input or lab system integration.")

    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

        with st.container():
            st.markdown("*<span class='notice important'>Important</span>: This app assists medical professionals in making a diagnosis, but should not be used as a substitute for professional diagnosis.*", unsafe_allow_html=True)

    with col2:

        add_predictions(input_data)

# Section 2: NucleiScan AI Chatbot
def nuclei_chatbot():
    st.subheader("NucleiScan AI: Chatbot")
    
    # Load chat history from file
    def load_chat_history():
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        return {}

    # Save chat history to file
    def save_chat_history(history):
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f)

    # Initialize session state
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = []
        st.session_state.chat_history = load_chat_history()
        st.session_state.active_chat = None

    col1, col2 = st.columns([4, 1])
    # Sidebar for chat history management
    with st.sidebar:
        st.header("Chat History")

        # Button to start new chat
        if st.button("➕ New Chat"):
            st.session_state.active_chat = None
            st.session_state.current_chat = []
            st.rerun()

        # Clear all history button
        if st.button("🗑️ Clear All History", type="primary"):
            st.session_state.chat_history = {}
            st.session_state.active_chat = None
            st.session_state.current_chat = []
            save_chat_history({})
            st.rerun()

        # Display saved chats with delete buttons
        for chat_id, chat_data in list(st.session_state.chat_history.items()):
            if st.button(
                f"💬 {chat_data['title'][:30]}...",
                key=f"btn_{chat_id}",
                use_container_width=True
            ):
                st.session_state.active_chat = chat_id
                st.session_state.current_chat = chat_data["messages"]
                st.rerun()
                
            if st.button(
                "clear", 
                key=f"del_{chat_id}",
                help="Delete this chat"
            ):
                del st.session_state.chat_history[chat_id]
                save_chat_history(st.session_state.chat_history)
                if st.session_state.active_chat == chat_id:
                    st.session_state.active_chat = None
                    st.session_state.current_chat = []
                st.rerun()

    with col1:

        for message in st.session_state.current_chat:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Generate chat responses
        def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
            for chunk in chat_completion:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                else:
                    yield ""

        # Chat input
        if prompt := st.chat_input("Ask a question..."):
            # Add user message to current chat
            st.session_state.current_chat.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.write(prompt)

            # Get AI response
            try:
                with st.spinner("Thinking..."):
                    chat_completion = client.chat.completions.create(
                        model="compound-beta",
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.current_chat
                        ],
                        max_tokens=MODEL_CONFIG['tokens'],
                        stream=True
                    )

                    with st.chat_message("assistant"):
                        chat_responses_generator = generate_chat_responses(chat_completion)
                        full_response = st.write_stream(chat_responses_generator)

                        # Add AI response to current chat
                        st.session_state.current_chat.append(
                            {"role": "assistant", "content": full_response}
                        )

                        # Save to history if this is a new chat
                        if st.session_state.active_chat is None:
                            chat_id = str(uuid.uuid4())  # Generate unique ID
                            st.session_state.chat_history[chat_id] = {
                                "title": prompt[:50],  # First 50 chars as title
                                "messages": st.session_state.current_chat
                            }
                            st.session_state.active_chat = chat_id
                            save_chat_history(st.session_state.chat_history)
                            st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Main function to run the Streamlit app
def main():
    st.set_page_config(
        page_title="NucleiScan AI",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )


    with open("../assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    # Call Section 1
    nuclei_predict()
    
    # Call Section 2
    nuclei_chatbot()                

if __name__ == '__main__':
    main() 