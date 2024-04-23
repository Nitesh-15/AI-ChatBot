import json
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from PIL import Image

logo = Image.open(r"D:\Machine Learning Projects\Medical Insurance Premium Predictor\STW-LOGO.png")
# st.image(logo)
col1, col2, col3 = st.columns(3)
with col1:
    st.write("")
with col2:
    st.image(logo, caption='STW Services')

with col3:
    st.write("")

intents = json.load(open('intents.json'))
tags = []
patterns = []

#  looping through all the intents and Identifying the patterns and greetings from the intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# to extract the features from the patterns
vector = TfidfVectorizer()
patterns_scaled = vector.fit_transform(patterns)

# Model
Bot = LogisticRegression(max_iter=100000)
Bot.fit(patterns_scaled, tags)


# Identifying the tag for the input
def ChatBot(input_message):
    input_message = vector.transform([input_message])
    pred_tag = Bot.predict(input_message)[0]
    for intent in intents['intents']:
        if intent['tag'] == pred_tag:
            response = random.choice(intent['responses'])
            return response


st.markdown("<h1 style='text-align: center; color: red;'>University AI ChatBot</h1>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = ChatBot(prompt)
    if response:
        response = f"AI ChatBot: {response}"  # Ensure the response starts with "AI ChatBot:"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
