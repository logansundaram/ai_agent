import os
from langchain_core.tools import Tool as CoreTool
from langchain_community.agent_toolkits.file_management.toolkit import FileManagementToolkit
from langchain_community.tools import DuckDuckGoSearchRun
from rag.retriever import build_initial_rag, add_new_file_to_rag

def get_tools(llm):
    tools = []

    # Web Search Tool
    search = CoreTool(
        name="Web Search",
        func=DuckDuckGoSearchRun().run,
        description="Search the web for current events or factual info"
    )

    # Calculator Tool
    def calculator(query: str) -> str:
        try:
            return str(eval(query))
        except Exception as e:
            return f"Error: {e}"

    calc_tool = CoreTool(
        name="Calculator",
        func=calculator,
        description="Evaluate basic math expressions"
    )

    # RAG Tool
    rag_tool = build_initial_rag(llm)

    # Root docs directory
    root_dir = os.path.abspath(os.path.join(os.getcwd(), "docs"))
    file_toolkit = FileManagementToolkit(root_dir=root_dir)
    file_tools = file_toolkit.get_tools()

    # Manually add working Write File tool with RAG update
    def wrapped_write_file(input_str):
        lines = input_str.strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines

        if not lines:
            return "Error: No input provided."

        filename = os.path.basename(lines[0])  # Prevent path traversal
        file_path = os.path.join(root_dir, filename)

        file_content = "\n".join(lines[1:]) if len(lines) > 1 else ""

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_content)

            rag_status = add_new_file_to_rag(file_path) if os.path.exists(file_path) else "File not found"
            return f"✅ File written to {file_path}\nContents:\n{file_content}\nRAG updated: {rag_status}"
        except Exception as e:
            return f"❌ Error writing file: {e}"


    write_tool = CoreTool(
        name="Write File",
        description=(
            "Write a file to the /docs folder.\n"
            "**Format:** First line is the filename (e.g., `colors.txt`), followed by the file content, each color on a new line.\n"
            "**Example Input:**\n"
            "colors.txt\\nred\\ngreen\\nblue\\nyellow\\npurple"
        ),
        func=wrapped_write_file
    )

    # Wrap remaining file tools (skip write if already defined)
    wrapped_file_tools = []
    for tool in file_tools:
        if tool.name.lower().startswith("write"):
            continue
        if hasattr(tool, "args") and isinstance(tool.args, dict) and len(tool.args) > 1:
            continue

        def make_wrapped_func(original_tool):
            return lambda x: original_tool.invoke(x)

        wrapped_tool = CoreTool(
            name=tool.name,
            description=tool.description,
            func=make_wrapped_func(tool)
        )
        wrapped_file_tools.append(wrapped_tool)

    # Final tool list
    tools.extend([rag_tool, search, calc_tool, write_tool] + wrapped_file_tools)
    return tools
