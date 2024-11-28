import os

from time import sleep

import openai
from dotenv import load_dotenv


def openai_chat_completion(
    messages,
    temperature=0,
    max_tokens=None,
    stop=None,
    n=None,
    model=None,
):
    """
    Wrapper function for creating chat completion request through OpenAI.

    Args:
        messages (list): A list of messages for the chat completion.
        functions (str, optional): A string representing the functions to be used.
        function_call (str, optional): A string representing the function call to be used.
        temperature (float, optional): A float representing the temperature for generating text. Defaults to 0.
        max_tokens (int, optional): An integer representing the maximum number of tokens. Defaults to None.
        stop (str, optional): A string representing the stopping condition for generating text. Defaults to None.
        n (int, optional): An integer representing the number of completions to generate. Defaults to None.
        model (str, optional): A string representing the model to be used. Defaults to "MODEL".

    Returns:
        ChatCompletion: An instance of the ChatCompletion class.
    """
    model = os.environ.get("MODEL", "Meta-Llama-3.1-8B-Instruct")
    json_data = {"model": model,  "messages": messages}
    
    if temperature is not None:
        json_data["temperature"] = temperature
    if max_tokens is not None:
        json_data["max_tokens"] = max_tokens
    if stop is not None:
        json_data["stop"] = stop
    if n is not None:
        json_data["n"] = n

    output = None
    while output is None:
        try:
            output = client.chat.completions.create(**json_data)
        except openai.OpenAIError as e:
            print(e)
            sleep(10)

    return output


if __name__ == "__main__":
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dotenv_path = os.path.join(project_dir, ".env")
    load_dotenv(dotenv_path)
    
    client = openai.OpenAI()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Please answer the following question."},
        {"role": "user", "content": "what is the capital of Canada?"}
    ]
    output = openai_chat_completion(messages=messages)
    print(output.choices[0].message.content)