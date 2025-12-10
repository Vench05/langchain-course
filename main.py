from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that search over internet.
    Args:
        query (str): The query to search for.
    Returns:
        The search results.
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)


def main():
    print("Hello from langchain-course!")
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    tools = [search]
    agent = create_agent(model=llm, tools=tools)

    content = "Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
    result = agent.invoke({"messages": [HumanMessage(content=content)]})
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
