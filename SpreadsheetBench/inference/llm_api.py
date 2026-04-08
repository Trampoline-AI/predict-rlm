from typing import List
from openai import OpenAI

def get_llm_response(messages: List[str], opt):
    client = OpenAI(api_key=opt.api_key, base_url=opt.base_url)
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": messages[i]} for i in range(len(messages))]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=opt.model,
    )
    return chat_completion.choices[0].message.content
