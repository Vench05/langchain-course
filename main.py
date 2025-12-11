from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()


class Source(BaseModel):
    """Schema for a source used by the agent."""

    url: str = Field(description="The URL of the source.")


class AgentResponse(BaseModel):
    """Schema for the agent response."""

    answer: str = Field(description="The Agent's answer to the user query.")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer."
    )


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
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

    content = "Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
    result = agent.invoke({"messages": [HumanMessage(content=content)]})
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
