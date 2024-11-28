import openai

from time import sleep

client = openai.OpenAI()

messages = [
    {
        "role": "system", "content": "You are a helpful assistant. Please answer the following question.",
        "role": "user", "content": "what is the capital of Canada?"
    }
]

output = None

while output is None:
    try:
        output = client.chat.completions.create(
            model="Meta-Llama-3.1-8B-Instruct",
            messages=messages,
        )
    
    except openai.APIError as e:
        print(e)
        sleep(10)
        
print(output.choices[0].message.content)