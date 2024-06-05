import os
import torch
from huggingface_hub import login
from langchain_community.vectorstores import FAISS
from utils import SentenceTransformerEmbeddings, create_db_from_files, search_top_k, load_llm__model
import streamlit as st
from utils_icon import icon


def main():
    login(token="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")
    # Load SentenceTransformer model
    # embedding_model = SentenceTransformerEmbeddings("BAAI/bge-m3")
    
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
                
                # create_db_from_files(data_path, vector_db_path, embedding_model)
                
                # Update the list of folders after creating a new folder
                list_of_folders = [folder for folder in os.listdir("database") 
                                if os.path.isdir(os.path.join("database", folder)) 
                                and not folder.endswith("_vector_db")]

        # Create a selectbox with the list of folders
        selected_folder = st.selectbox("Select a folder", list_of_folders)
     
        
    # db = FAISS.load_local(vector_db_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
    
    # tokenizer, model = load_llm__model("Viet-Mistral/Vistral-7B-Chat")
    
    # system_prompt = "Tôi là một trợ lí Tiếng Việt nhiệt tình và trung thực. Tôi luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
    # conversation = [{"role": "system", "content": system_prompt}]
    
    # st.title("Chatbox with RAG")
    # st.write("Xin chào, tôi là trợ lý chatbot của bạn. Tôi có thể giúp gì cho bạn không?")
    
    # if 'generated' not in st.session_state:
    #     st.session_state['generated'] = []

    # if 'past' not in st.session_state:
    #     st.session_state['past'] = []
    
    # def get_text():
    #     input_text = st.text_input("You: ", key="input")
    #     return input_text
    
    # user_input = get_text()
    
    # if user_input:
    #     conversation.append({"role": "user", "content": user_input})
    #     input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
        
    #     context = ""
    #     if user_input[-1] == '?':
    #         context = '\n'.join(x.page_content for x in search_top_k(db, embedding_model, user_input, 3))
    #         conversation = [{"role": "system", "content": f'Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n {context}'}]
    #         conversation.append({"role": "user", "content": user_input})
        
    #     out_ids = model.generate(
    #         input_ids=input_ids,
    #         max_new_tokens=768,
    #         do_sample=True,
    #         top_p=0.9,
    #         top_k=10,
    #         temperature=0.1,
    #         repetition_penalty=1.05,
    #         pad_token_id=tokenizer.eos_token_id,
    #     )
    #     assistant_response = tokenizer.batch_decode(out_ids[:, input_ids.size(1):], skip_special_tokens=True)[0].strip()
    #     conversation.append({"role": "assistant", "content": assistant_response})
        
    #     st.session_state.past.append(user_input)
    #     st.session_state.generated.append(assistant_response)
    
    # if st.session_state['generated']:
    #     for i in range(len(st.session_state['generated']) - 1, -1, -1):
    #         message(st.session_state["generated"][i], key=str(i))
    #         message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

if __name__ == "__main__":
    main()