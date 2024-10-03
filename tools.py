from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv
import yaml

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

load_dotenv()

# Retrieve the TAVILY_API_KEY from the environment variables
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialize tools based on configuration
tools = []
for tool_cfg in config['tools']:
    if tool_cfg['name'] == 'TavilySearchResults':
        tavily_tool = TavilySearchResults(max_results=tool_cfg.get('max_results', 5))
        tools.append(tavily_tool)
    elif tool_cfg['name'] == 'PythonREPL':
        python_repl = PythonREPL()
        repl_tool = Tool(
            name="python_repl",
            description=(
                "A Python shell. Use this to execute python commands, particularly complex math equations. "
                "Input should be a valid python command. If you want to see the output of a value, you should "
                "print it out with `print(...)`. ONLY USE THIS FOR COMPLEX MATH EQUATIONS. For simpler ones "
                "that you can do yourself, you do not need to use this."
            ),
            func=python_repl.run,
        )
        tools.append(repl_tool)
    # Add more tools as needed
    else:
        raise ValueError(f"Unknown tool {tool_cfg['name']}")

# Initialize a ToolNode with the list of tools
tool_node = ToolNode(tools)