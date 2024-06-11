import os
import torch
import time
from huggingface_hub import login
from langchain_community.vectorstores import FAISS
from vector_database import SentenceTransformerEmbeddings, create_db_from_files, search_top_k, make_dbs
# from llm import load_llm__model, llm_answering
from gemini import gemini, gemini_answering
import streamlit as st
from streamlit_chatbox import *


embedding_model = None

def list_folders(folder_base: str):
    return [folder for folder in os.listdir(folder_base) 
                   if os.path.isdir(os.path.join(folder_base, folder)) 
                   and not folder.endswith("_vector_db")]

def main(): 
    # config sidebar
    list_of_folders = list_folders("database")

    with st.sidebar:
        st.info("**Upload your documents... ↓**", icon="👋🏾")
        uploaded_files = st.file_uploader("Upload files", type=["txt", "pdf", "docx"], accept_multiple_files=True)
        folder_name = st.text_input("Enter a name for the folder")
        submitted = st.button("Submit")

        if submitted and uploaded_files is not None and folder_name:
            if folder_name in list_of_folders:
                dir_path = f"database/{folder_name}"
                #copunt number folder in dir_path
                subfolder_count = len([name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))])
                data_path = f"database/{folder_name}/{subfolder_count}_fol"
                vector_db_path = f"database/{folder_name}_vector_db/{subfolder_count}_fol"
            else:
                subfolder_name = "0_fol"
                data_path = f"database/{folder_name}/{subfolder_name}"
                vector_db_path = f"database/{folder_name}_vector_db/{subfolder_name}"
                
            # Create the database folder if it doesn't exist
            os.makedirs(data_path, exist_ok=True)
            os.makedirs(vector_db_path, exist_ok=True)
            
            # Save the uploaded files to the database folder
            for file in uploaded_files:
                file_path = os.path.join(data_path, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
            create_db_from_files(data_path, vector_db_path, embedding_model)
            
            # Update the list of folders after creating a new folder
            list_of_folders = list_folders("database")

        st.divider()
        # Create a selectbox with the list of folders
        list_of_folders.insert(0, "-")
        selected_folder = st.selectbox("Select a folder", list_of_folders, index=0)
    
    #load vector db
    if selected_folder != "-":
        folder_path = f"database/{selected_folder}_vector_db"
        dbs = make_dbs(folder_path, embedding_model)
        
    #chatbox
    chat_box = ChatBox()
    chat_box.use_chat_name("chat1")
    chat_box.init_session()
    chat_box.output_messages()
    
    if user_input := st.chat_input('input your question here'):
        chat_box.user_say(user_input)
        # conversation.append({"role": "user", "content": user_input})
        time.sleep(1)
        context = ""
        if user_input[-1] == '?':
            context = '\n'.join(x.page_content for x in search_top_k(dbs, embedding_model, user_input, 3))

        #answering
        # assistant_response = llm_answering(model, tokenizer, user_input, conversation)
        assistant_response = gemini_answering(model, user_input, context)
    
        elements = chat_box.ai_say(
            [
                # you can use string for Markdown output if no other parameters provided
                Markdown("thinking", in_expander=False,
                            expanded=True, title="answer"),
            ]
        )
        time.sleep(1)
        text = ""
        for x in assistant_response:
            text += x
            chat_box.update_msg(text, element_index=0, streaming=True)
            time.sleep(0.05)
        
embedding_model = SentenceTransformerEmbeddings("BAAI/bge-m3")
model =  gemini()
st.title("Chatbox with RAG")
st.write("Xin chào, tôi là trợ lý chatbot của bạn. Tôi có thể giúp gì cho bạn không?")

if __name__ == "__main__":
    main()
