import google.generativeai as genai
import time

def gemini():
    genai.configure(api_key="AIzaSyDtOukKpgHvIwkvwPaCOC4gcIw7CE8VtB0")
    # Set up the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
    }
    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCk_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCk_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCk_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCk_NONE"
    },
    ]
    
    system_instruction = "bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Bạn luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
    
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                generation_config=generation_config,
                                safety_settings=safety_settings,
                                system_instruction=system_instruction)
    convo = model.start_chat(history=[
    {
        "role": "user",
        "parts": ["hey"]
    },
    {
        "role": "model",
        "parts": ["Hey! How can I help you today? 😊"]
    },
    ])
    return convo

def gemini_answering(model, prompt, context):
    query = f"Để trả lời câu hỏi {prompt}, hãy dùng thông tin sau: {context} . Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.\n "
    # print(query)
    return model.send_message(query).text

if __name__ == "__main__":
    model = gemini()
    prompt = "ai là tác giả của bài hát abc"

    context = "abc dược sáng tác bởi ca sĩ người miền nam. Bài hát này đã được phát hành vào năm 2020. Bài hát này đã đạt được nhiều giải thưởng lớn. Bài hát này đã được phát hành bởi công ty âm nhạc lớn nhất Việt Nam. chủ nhan của nó tên là tuấn"

    print(gemini_answering(model, prompt, context))