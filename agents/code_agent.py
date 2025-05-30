from langchain.agents import tool
from langchain.tools import Tool
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

@tool
def generate_code(prompt: str) -> str:
    """Generates code based on the prompt."""
    chain = (
        ChatPromptTemplate.from_template("Generate Python code for: {prompt}")
        | ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")
        | StrOutputParser()
    )
    return chain.invoke({"prompt": prompt})

code_tool = Tool(
    name="CodeTool",
    func=generate_code,
    description="Use this tool to generate Python code from natural language"
)
