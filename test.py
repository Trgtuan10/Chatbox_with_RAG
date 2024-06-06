import streamlit as st
from streamlit_chatbox import *
import time
import simplejson as json
import os

st.title("Chatbox with RAG")
llm = FakeLLM()
chat_box = ChatBox()
chat_box.use_chat_name("chat1") # add a chat conversatoin

list_of_folders = [folder for folder in os.listdir("database") 
                   if os.path.isdir(os.path.join("database", folder)) 
                   and not folder.endswith("_vector_db")]

with st.sidebar:
        st.info("**Upload your documents... ↓**", icon="👋🏾")
        uploaded_files = st.file_uploader("Upload files", type=["txt", "pdf", "docx"], accept_multiple_files=True)
        folder_name = st.text_input("Enter a name for the folder")
        submitted = st.button("Submit")

        if submitted and uploaded_files is not None and folder_name:
            if folder_name in list_of_folders:
                st.warning("A folder with the same name already exists. Please choose a different name.")
            else:
                data_path = f"database/{folder_name}"
                vector_db_path = f"database/{folder_name}_vector_db"
                # Create the database folder if it doesn't exist
                os.makedirs(data_path, exist_ok=True)
                
                # Save the uploaded files to the database folder
                for file in uploaded_files:
                    file_path = os.path.join(data_path, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getvalue())
                
                # Update the list of folders after creating a new folder
                list_of_folders = [folder for folder in os.listdir("database") 
                                if os.path.isdir(os.path.join("database", folder)) 
                                and not folder.endswith("_vector_db")]

        st.divider()
        # Create a selectbox with the list of folders
        selected_folder = st.selectbox("Select a folder", list_of_folders, index=0)


chat_box.init_session()
chat_box.output_messages()

if query := st.chat_input('input your question here'):
    chat_box.user_say(query)
    generator = llm.chat_stream(query)
    elements = chat_box.ai_say(
        [
            # you can use string for Markdown output if no other parameters provided
            Markdown("thinking", in_expander=False,
                        expanded=True, title="answer"),
        ]
    )
    time.sleep(1)
    text = ""
    for x, docs in generator:
        text += x
        chat_box.update_msg(text, element_index=0, streaming=True)
        
