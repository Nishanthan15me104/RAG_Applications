from agents.main_agent import agent_executor
from dotenv import load_dotenv
load_dotenv()

# You could ask high-level questions here
if __name__ == "__main__":
    response = agent_executor.run("Summarize this text: Transformers are sequence models ...")
    print("Agent Response:", response)


    response2 = agent_executor.run("Write a Python function to scrape job listings from LinkedIn.")
    print("Code Generation:", response2)
