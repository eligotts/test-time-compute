from langchain_core.messages import (
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
import functools
from models import llm_mapping, config 
from tools import tools

def create_response_agent(llm, tools):
    """
    Create an agent for generating initial responses to complex questions.

    Args:
        llm: The language model to use for generating responses.
        tools: A list of tools that the agent can use to help answer questions.

    Returns:
        A prompt template bound to the language model and tools.
    """

    # load and read the text from file at prompt/initial_response.txt
    with open('prompts/response.txt', 'r') as file:
        data = file.read()

    system_prompt = data + "\n To help answer your question, you have access to the following tools: {tool_names} \n"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

def create_summary_agent(llm):
    """
    Create an agent for summarizing complex chains of reasoning.

    Args:
        llm: The language model to use for generating summaries.

    Returns:
        A prompt template bound to the language model.
    """

    with open('prompts/summary.txt', 'r') as file:
        system_prompt = file.read()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt | llm

def create_revision_agent(llm, tools):
    """
    Create an agent for revising pre-generated answers to questions.

    Args:
        llm: The language model to use for generating revised responses.
        tools: A list of tools that the agent can use to help revise answers.

    Returns:
        A prompt template bound to the language model and tools.
    """

    with open('prompts/reviser.txt', 'r') as file:
        data = file.read()

    system_prompt = data + "\n To help, you have access to the following tools: {tool_names} \n"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
                
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

def agent_node(state, agent, name):
    """
    Helper function to create a node for a given agent.

    Args:
        state: The current state of the conversation.
        agent: The agent to invoke.
        name: The name of the agent.

    Returns:
        A dictionary representing the updated state after invoking the agent.
    """
    result = agent.invoke(state)
    # Convert the agent output into a format that is suitable to append to the global state
    if name == "Summary":
        return {
            "final_answer": result.content,
        }
      
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Track the sender to know who to pass to next in the workflow.
        "sender": name,
    }

# Create initial response agents and nodes
initial_response_agents = {}
revision_agents = {}
for agent_cfg in config['llms']['response_agents']:
    name = agent_cfg['name']
    llm = llm_mapping["response_agents"][f'{name}']

    response_agent = create_response_agent(llm, tools)
    agent_node_fn = functools.partial(agent_node, agent=response_agent, name=name)
    initial_response_agents[name] = agent_node_fn

    revision_agent = create_revision_agent(llm, tools)
    agent_node_fn_rev = functools.partial(agent_node, agent=revision_agent, name=name)
    revision_agents[name] = agent_node_fn_rev

# Do the same for answer_summary agent
answer_summary_llm = llm_mapping['answer_summary_agent']
answer_summary_agent = create_summary_agent(answer_summary_llm)
answer_summary_node = functools.partial(agent_node, agent=answer_summary_agent, name='Summary')
