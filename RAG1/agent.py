from dotenv import load_dotenv
from .prompts import system_agent_prompt
from .tools import collecting_info
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import HumanMessage
from langchain_groq import ChatGroq
from .objects import Answer
import RAG1.tools as tl 

load_dotenv()

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

agent = create_agent(
    model=llm,
    tools=[collecting_info],
    system_prompt=system_agent_prompt,
    # Groq rejects JSON mode combined with tool calls, so force tool-based structured output.
    response_format=ToolStrategy(Answer),
)

def ask(question) -> dict:
    """Answer the user's question about the PDF"""

    tl.last_retrieved_context = []

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    structured_response = result["structured_response"]

    # context is what the retriever really returned, not text the model wrote
    return {
        "answer": structured_response.answer,
        "context": list(tl.last_retrieved_context)
    }

# print(ask("what is the diet of penguin?"))