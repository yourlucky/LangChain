from gpt4all import GPT4All

from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate


model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf") # downloads / loads a 4.66GB LLM
with model.chat_session():
    print(model.generate("How can I run LLMs efficiently on my laptop?", max_tokens=1024))