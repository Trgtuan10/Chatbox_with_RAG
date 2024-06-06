import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from vector_database import search_top_k


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

def llm_answering(model, tokenizer, user_input, conversation):
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
    assistant_response = tokenizer.batch_decode(out_ids[:, input_ids.size(1):], skip_special_tokens=True)[0].strip()
    return assistant_response