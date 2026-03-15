import uuid
from datetime import datetime

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
import gradio as gr

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

def get_date():
    """
    Get the current date
    """
    return datetime.now().strftime("%Y-%m-%d")

conn = sqlite3.connect('chat_memory.db', check_same_thread=False)
checkpointer = SqliteSaver(conn)

system_prompt = """
You are a helpful assistant.
Answer all user's question
Use the get_date function if the user asks about today's date.
"""

llm = ChatOllama(model="qwen3.5:4b")

agent =create_agent(
    model=llm,
    tools=[get_date],
    system_prompt=system_prompt,
    checkpointer=checkpointer)

def chat(message, history, thread_id):
    config = {"configurable":{"thread_id": thread_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config
    )
    last_response = response["messages"][-1].content
    return last_response


with gr.Blocks() as demo:
    gr.Markdown("# AI Chatbot")
    thread_id = gr.State(value=lambda: str(uuid.uuid4()))
    gr.ChatInterface(fn=chat, additional_inputs=[thread_id], title="Chat with AI")

demo.launch(share=True)