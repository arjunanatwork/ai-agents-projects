from datetime import datetime

from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_date():
    """
    Get the current date
    """
    return datetime.now().strftime("%Y-%m-%d")

system_prompt = """
Use the get_date function to get the current date.
"""

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

agent =create_agent(model=llm, tools=[get_date], system_prompt=system_prompt)
user_question = input("Enter your question: ")
response = agent.invoke({"messages": [{ "role": "user", "content": user_question}]})

print(response['message'][-1].content[0]['text'])
