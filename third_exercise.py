import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from ollama import Client

load_dotenv()

st.set_page_config(
    page_title="Genie",
    page_icon="🔮",
    layout="centered",
)

st.title("Genie 🧞‍♂️")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.header("Model settings")
provider = st.sidebar.radio(
    "Choose an LLM provider",
    ["Groq", "Ollama"],
)

model_options = {
    "Groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ],
    "Ollama": [
        "gemma4:31b",
        "gpt-oss:120b",
    ],
}
selected_model = st.sidebar.selectbox(
    "Choose a model",
    model_options[provider],
)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if provider == "Groq":
    llm = ChatGroq(
        model=selected_model,
        temperature=0.0,
    )
else:
    llm = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
)

user_prompt = st.chat_input("Ask Genie...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    ollama_messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        *st.session_state.chat_history
    ]

    with st.spinner("Genie is thinking..."):
        if provider == "Groq":
            response = llm.invoke(ollama_messages)
        else:
            response = llm.chat(selected_model, messages=ollama_messages)

    if provider == "Groq":
        assistant_response = response.content
    else:
        assistant_response = response['message']['content']

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
