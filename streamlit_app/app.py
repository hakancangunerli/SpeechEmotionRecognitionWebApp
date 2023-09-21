import streamlit as st
import openai
import speech_recognition as sr
import os
openai.api_type = ""
openai.api_base = ""
openai.api_version = ""
openai.api_key= ""
# Streamlit UI
st.title("Speech to Text and OpenAI Converter")
st.markdown("This web app converts your speech into text and sends it to OpenAI for responses.")
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
def sentiment(text):
    from transformers import pipeline 
    sentiment_pipeline = pipeline("sentiment-analysis")
    label = sentiment_pipeline(text)
    return label[0]["label"]


# Function to convert speech to text
def speech_to_text():
    # first record the audio from recording.py
    from recording import record
    record()
     
    recognizer = sr.Recognizer()
    st.info("Please speak into your microphone...")
    audio_file_path = "output.wav"  # Change this path to your audio file
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"Recognized as: {text}")
            # tag the conversation as either positive or a negative conversation piece.
            # add the code from the main()
            # now do the emotion prediction 
            from model_prediction import predict
            prediction = predict(audio_file_path)
            st.success(f"Emotion: {prediction}, Sentiment: {sentiment(text)}")
            send_message_to_openai(text)
            

        except sr.UnknownValueError:
            st.warning("Could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results: {e}")
        # delete the audio recording file 
            

# Send the user's message to OpenAI
def send_message_to_openai(text):
    st.session_state["messages"].append({"role": "user", "content": text})
    response_openai = generate_response(st.session_state["messages"])
    st.session_state["messages"].append({"role": "assistant", "content": response_openai})

# Generate a new response if the last message is not from the assistant
def generate_response(messages):
    # faiss_answer = faiss_search(messages)
    content_text = ""
    # content_text += faiss_answer
    conversation = [
        {"role": "system", "content":content_text},
    ]
    conversation.extend(messages)
    response = openai.ChatCompletion.create(
        engine="", # Replace with your actual engine ID 
        messages=conversation,
        temperature=0,
        max_tokens=800,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    
    return response['choices'][0]['message']['content']

# Streamlit UI


if st.button("Start Recording"):
    speech_to_text()

# Display chat messages
for message in st.session_state["messages"]:
    if message["role"] == "user":
        if "Based on the context" in message["content"]:
            displayed_content = message["content"].replace("Based on the context, ", "")
            with st.chat_message("user"):
                st.write(displayed_content)
        else:
            with st.chat_message("user"):
                st.write(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(message["content"])