import os
import torch
import time
from huggingface_hub import login
from langchain_community.vectorstores import FAISS
from vector_database import SentenceTransformerEmbeddings, create_db_from_files, search_top_k
from llm import load_llm__model, llm_answering
import streamlit as st
from streamlit_chatbox import *

embedding_model = None
tokenizer, model = None, None

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
                create_db_from_files(data_path, vector_db_path, embedding_model)
                
                # Update the list of folders after creating a new folder
                list_of_folders = list_folders("database")

        st.divider()
        # Create a selectbox with the list of folders
        list_of_folders.insert(0, "-")
        selected_folder = st.selectbox("Select a folder", list_of_folders, index=0)
    
    #load vector db
    if selected_folder != "-":
        vector_db_path = f"database/{selected_folder}_vector_db"
        db = FAISS.load_local(vector_db_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
        
    #chatbox
    chat_box = ChatBox()
    chat_box.use_chat_name("chat1")
    chat_box.init_session()
    chat_box.output_messages()
    
    system_prompt = "Tôi là một trợ lí Tiếng Việt nhiệt tình và trung thực. Tôi luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
    conversation = [{"role": "system", "content": system_prompt}]
    
    if user_input := st.chat_input('input your question here'):
        chat_box.user_say(user_input)
        conversation.append({"role": "user", "content": user_input})
    
        context = ""
        if user_input[-1] == '?':
            context = '\n'.join(x.page_content for x in search_top_k(db, embedding_model, user_input, 3))
            conversation = [{"role": "system", "content": f'Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n {context}'}]
            conversation.append({"role": "user", "content": user_input})

        #answering
        assistant_response = llm_answering(model, tokenizer, user_input, conversation)
    
        time.sleep(1)
        text = ""
        elements = chat_box.ai_say(
            [
                Markdown("thinking", in_expander=False,
                         expanded=True, title="answer"),
            ]
        )
        for x in assistant_response:
            text += x
            chat_box.update_msg(text, element_index=0, streaming=True)
        

if __name__ == "__main__":
    login(token="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")
    # Load model
    embedding_model = SentenceTransformerEmbeddings("BAAI/bge-m3")
    tokenizer, model = load_llm__model("Viet-Mistral/Vistral-7B-Chat")
    
    st.title("Chatbox with RAG")
    st.write("Xin chào, tôi là trợ lý chatbot của bạn. Tôi có thể giúp gì cho bạn không?")
    main()
