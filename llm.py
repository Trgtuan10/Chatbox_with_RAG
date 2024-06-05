import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from vector_database import  search_top_k

def load_llm__model(model_name: str):
    lora_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # load_in_8bit=True,
    )

    # model_name = 'Viet-Mistral/Vistral-7B-Chat'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=lora_config,
    #     torch_dtype=torch.bfloat16, # change to torch.float16 if you're using V100
        device_map="auto",
        use_cache=True,
    )
    return tokenizer, model
    
def answering_by_llm(db):
    tokenizer, model = answering_by_llm("Viet-Mistral/Vistral-7B-Chat")
    
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
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)

        # Find appropreate context
        context = ""
        if human[-1] == '?':
            context = '\n'.join(x.page_content for x in search_top_k(db, human, 3))
            print(context)
            # print("context: ", context)
            # Add context to conversation
            # conversation.append({"role": "assistant", "content": f"Dưới đây là một vài văn bản có thể chứa thông tin liên quan đến câu hỏi của bạn: {context}" })
            # conversation.append({"role": "user", "content": f'Vậy dựa vào đoạn văn bản trên mà bạn đã tìm được, hãy cho tôi biết {human}' })

            conversation = [{"role": "system", "content": f'Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n {context}'}]
            conversation.append({"role": "user", "content": human })

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
