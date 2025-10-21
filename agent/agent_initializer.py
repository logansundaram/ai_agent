from agent.tools import get_tools
from agent.memory import get_memory
from langchain.agents import initialize_agent, AgentType
from langchain_ollama.llms import OllamaLLM

def create_agent():
    llm = OllamaLLM(model="gemma3:27b")
    tools = get_tools(llm)
    memory = get_memory()
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent
