import streamlit as st

with st.sidebar:
    st.title("Settings")
    st.caption("🔑 Press here to up load your documents")
    #upload multi file here
    file_upload = st.file_uploader("Upload your documents", type=["txt"], accept_multiple_files=True)
    #a button to accept
    if st.button("Upload"):
        st.write("Files uploaded successfully")
st.title("💬 Chatbot")
st.caption("🚀 Chatbox to answer about your documentations")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)