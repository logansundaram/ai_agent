from agent.tools import get_tools
from agent.memory import get_memory
from langchain.agents import initialize_agent, AgentType #deprected
from langchain_ollama.llms import OllamaLLM #deprected

def create_agent():
    llm = OllamaLLM(model="gpt-oss:20b")
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
