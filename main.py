from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


def main():
    print("Hello from langchain-course!")
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools)

    content = "Search for 3 job postings for an ai engineer using langchain in the Philippines on linkedin and list their details"
    result = agent.invoke({"messages": [HumanMessage(content=content)]})
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
