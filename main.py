import os
import torch
from huggingface_hub import login
from langchain_community.vectorstores import FAISS

from vector_database import SentenceTransformerEmbeddings, create_db_from_files, search_top_k, load_llm__model

def main():
    login(token="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")
    # Load SentenceTransformer model
    embedding_model = SentenceTransformerEmbeddings("BAAI/bge-m3")

    # Create vector database
    data_path = "data"
    vector_db_path = "vector_db"
    if not os.path.exists(vector_db_path):
        db = create_db_from_files(data_path, vector_db_path, embedding_model)
    else:
        db = FAISS.load_local(vector_db_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
    
    tokenizer, model = load_llm__model("Viet-Mistral/Vistral-7B-Chat")
    
    system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
    system_prompt += "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực."
    system_prompt += "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trả lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."

    system_prompt = "Tôi là một trợ lí Tiếng Việt nhiệt tình và trung thực. Tôi luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
    # system_prompt += "Câu trả lời của tôi không chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Tôi luôn đảm bảo rằng các câu trả lời của tôi không có thiên kiến xã hội và mang tính tích cực."
    # system_prompt += "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, tôi sẽ giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu tôi không biết câu trả lời cho một câu hỏi, tôi sẽ trả lời là tôi không biết và không chia sẻ thông tin sai lệch."

    conversation = [{"role": "system", "content": system_prompt }]

    print("Xin chào, tôi là trợ lý chatbot của bạn. Tôi có thể giúp gì cho bạn không?")

    conversation = []

    while True:
        human = input("You: ")
        if human.lower() == "reset":
            # conversation = [{"role": "system", "content": system_prompt }]
            print("The chat history has been cleared!")
            continue
        elif human.lower() == "quit":
            break

        conversation.append({"role": "user", "content": human })

        # Find appropreate context
        context = ""
        if human[-1] == '?':
            context = '\n'.join(x.page_content for x in search_top_k(db, embedding_model, human, 3))
            # print(context)

            conversation = [{"role": "system", "content": f'Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n {context}'}]
            conversation.append({"role": "user", "content": human })

        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
        out_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=768,
            do_sample=True,
            top_p=0.9,
            top_k=10,
            temperature=0.1,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
        assistant = tokenizer.batch_decode(out_ids[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip()
        print("Assistant: ", assistant)
        conversation.append({"role": "assistant", "content": assistant })

if __name__ == "__main__":
    main()
