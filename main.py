from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.agents import AgentState
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent

class Source(BaseModel):
    """Schema for a source used by the agent."""

    url: str = Field(description="The URL of the source.")


class AgentResponse(BaseModel):
    """Schema for the agent response."""

    answer: str = Field(description="The Agent's answer to the user query.")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer."
    )



def main():
    print("Hello from langchain-course!")
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    tools = [TavilySearch()]
    react_propmt = hub.pull('hwchase17/react')
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_propmt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    chain = agent_executor

    content = "Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
    # result = agent.invoke({"input": [HumanMessage(content=content)]})
    result = chain.invoke({"input": content})
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
