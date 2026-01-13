from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.tools import tool
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from prompt import REACT_PROPMT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse



def main():
    print("Hello from langchain-course!")
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    tools = [TavilySearch()]
    structured_llm = llm.with_structured_output(AgentResponse)
    react_propmpt_with_format_instructions = PromptTemplate(
        template=REACT_PROPMT_WITH_FORMAT_INSTRUCTIONS,
        input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
        ).partial(format_instructions="")
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_propmpt_with_format_instructions)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    extract_output = RunnableLambda(lambda x: x["output"])
    chain = agent_executor | extract_output | structured_llm

    content = "Search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
    # result = agent.invoke({"input": [HumanMessage(content=content)]})
    result = chain.invoke({"input": content})
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
