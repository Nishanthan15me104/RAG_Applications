from langchain.agents import initialize_agent, AgentType
from agents.summarize_agent import summarize_tool
from agents.code_agent import code_tool
# You can import more tools (vision_tool, search_tool, etc.)

from langchain_groq import ChatGroq

llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")

tools = [summarize_tool, code_tool]  # Add vision_tool, etc. here

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
