from dataclasses import dataclass

import yfinance as yf
from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_community.tools import YahooFinanceNewsTool
from langchain_core.tools import tool

load_dotenv(find_dotenv(), override=True)

#Initialize the model
model = init_chat_model("gpt-4o")

#Define the schema for the response
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    currentNews: str | None = None
    stockPrice: float | None = None

# ✅ New tool: fetches the real-time stock price
@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol (e.g. MSFT, AAPL)."""
    stock = yf.Ticker(ticker)
    price = stock.fast_info.last_price
    if price is None:
        return f"Could not retrieve price for {ticker}."
    return f"The current price of {ticker} is ${price:.2f}"

#Define the Yahoo Finance Tool
tools=[YahooFinanceNewsTool(), get_stock_price]

agent = create_agent(
    model=model,
    system_prompt="""
         You are a financial assistant that will help understand financial news about a particular stock.
         Answer the user query and use necessary tools.
         """,
    response_format=ToolStrategy(ResponseFormat),
    tools=tools
)

query = input("What can i help you with? ")

response = agent.invoke(
    {"messages": [{"role": "user", "content": query}]},
)

print(response['structured_response'])

