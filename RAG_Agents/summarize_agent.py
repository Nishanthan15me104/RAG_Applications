from langchain.agents import tool
from langchain.tools import Tool
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Define a simple summarization tool
@tool
def summarize_text(text: str) -> str:
    """Summarizes a given text or table."""
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following content:\n\n{content}"
    )
    model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"content": text})

summarize_tool = Tool(
    name="SummarizeTool",
    func=summarize_text,
    description="Use this for summarizing any long text or table"
)
