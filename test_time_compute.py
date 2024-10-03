from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, START
import os
import time

from state import AgentState, GraphState
from response_agents import answer_summary_node, initial_response_agents, revision_agents

from tools import tool_node
from upper_agents import (
    ask_question, join_graph, get_info_for_initial_response,
    get_info_for_revision_response, difficulty_agent, commenter_agent,
    scorer_agent, check_done_agent, final_summary_agent, beam_search_agent,
    initial_response_handler, revised_response_handler
)

from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.key_binding import KeyBindings


# ROUTERS

def router_tools(state) -> Literal["call_tool", "__end__"]:
    """
    Determines whether a tool should be called or if the process should end based on the state.
    
    Args:
        state (dict): The current state of the agent.
        
    Returns:
        Literal["call_tool", "__end__"]: The next step in the workflow.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "call_tool"
    
    return "__end__"


def initial_response_router(state) -> Literal["get_initial_response", "difficulty_assessment", "beam_search_agent"]:
    """
    Routes the workflow for the initial response generation.

    Args:
        state (dict): The current state of the agent.

    Returns:
        Literal["get_initial_response", "difficulty_assessment", "beam_search_agent"]: The next step.
    """
    start = state["start"]
    responses = state["responses"]
    threads = state["threads"]

    if len(responses) < threads:
        return "get_initial_response"
    elif start:
        return "difficulty_assessment"

    return "beam_search_agent"


def difficulty_router(state) -> Literal["get_initial_response", "beam_search_agent"]:
    """
    Routes the workflow for the difficulty assessment process.

    Args:
        state (dict): The current state of the agent.

    Returns:
        Literal["get_initial_response", "beam_search_agent"]: The next step.
    """
    responses = state["responses"]
    threads = state["threads"]

    if len(responses) < threads:
        return "get_initial_response"

    return "beam_search_agent"


def revision_router(state) -> Literal["check_done", "get_revision_response", "summary"]:
    """
    Routes the workflow for the revision process.

    Args:
        state (dict): The current state of the agent.

    Returns:
        Literal["check_done", "get_revision_response", "summary"]: The next step.
    """
    index = state["index"]
    threads = state["threads"]
    revisions = state["revisions"]
    responses = state["responses"]
    
    if len(responses[-1]["content"]) == revisions + 1:
        return "summary"
    elif index == threads:
        return "check_done"

    return "get_revision_response"


def scorer_router(state) -> Literal["initial_response_handler", "revised_response_handler"]:
    """
    Routes the workflow for scoring the responses.

    Args:
        state (dict): The current state of the agent.

    Returns:
        Literal["initial_response_handler", "revised_response_handler"]: The next step.
    """
    start = state["start"]
    threads = state["threads"]
    responses = state["responses"]

    if start or len(responses) < threads:
        return "initial_response_handler"
    
    return "revised_response_handler"


def done_router(state) -> Literal["continue", "__end__"]:
    """
    Routes the workflow to determine if the process is complete or should continue.

    Args:
        state (dict): The current state of the agent.

    Returns:
        Literal["continue", "__end__"]: The next step.
    """
    if state["done"]:
        return "__end__"
    
    return "continue"


# WORKFLOW GRAPH SETUP

# Create the initial response graph
initial_response_workflow = StateGraph(AgentState)

# Add nodes to the initial response workflow
initial_response_workflow.add_node("call_tool", tool_node)
initial_response_workflow.add_node("Summary", answer_summary_node)

# Add all initial response agents to the workflow
for name, node in initial_response_agents.items():
    initial_response_workflow.add_node(name, node)
    
    # Add conditional edges for agents and tool usage
    initial_response_workflow.add_conditional_edges(
        name,
        router_tools,
        {"call_tool": "call_tool", "__end__": "Summary"},
    )

# Store agent names in a dictionary
name_dict = {name: name for name in initial_response_agents.keys()}

# Add conditional edges for tool invocation
initial_response_workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    name_dict,
)

# Add the starting edge to invoke the appropriate agent based on the sender
initial_response_workflow.add_conditional_edges(
    START,
    lambda x: x["sender"],
    name_dict,
)

# Add the final summary node and compile the graph
initial_response_workflow.add_edge("Summary", END)
graph_initial = initial_response_workflow.compile()


def enter_chain(message_and_agent):
    """
    Kick off the initial response chain by setting up the starting state.
    
    Args:
        message_and_agent (tuple): A tuple containing the message and the agent.
        
    Returns:
        dict: The initial state for the workflow.
    """
    message, agent = message_and_agent
    return {
        "messages": [HumanMessage(content=message)],
        "sender": agent,
    }


# Combine the initial response chain with the compiled graph
initial_response_chain = enter_chain | graph_initial


# CREATE REVISION GRAPH

revision_workflow = StateGraph(AgentState)

# Add nodes to the revision workflow
revision_workflow.add_node("call_tool", tool_node)
revision_workflow.add_node("Summary", answer_summary_node)

# Add all revision agents to the workflow
for name, node in revision_agents.items():
    revision_workflow.add_node(name, node)
    
    # Add conditional edges for agents and tool usage
    revision_workflow.add_conditional_edges(
        name,
        router_tools,
        {"call_tool": "call_tool", "__end__": "Summary"},
    )

# Add conditional edges for tool invocation
revision_workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    name_dict,
)

# Add the starting edge to invoke the appropriate agent based on the sender
revision_workflow.add_conditional_edges(
    START,
    lambda x: x["sender"],
    name_dict,
)

# Add the final summary node and compile the graph
revision_workflow.add_edge("Summary", END)
graph_revision = revision_workflow.compile()


def enter_chain_revision(question_and_agent_and_previous_response_and_comments):
    """
    Kick off the revision chain by setting up the starting state.
    
    Args:
        question_and_agent_and_previous_response_and_comments (tuple): A tuple containing the question, agent, previous response, and comments.
        
    Returns:
        dict: The initial state for the workflow.
    """
    question, agent, previous_response, comments = question_and_agent_and_previous_response_and_comments
    message = f"Here is the question: {question}\nHere is the previous response: {previous_response}\nHere are the comments: {comments}\n"
    
    return {
        "messages": [HumanMessage(content=message)],
        "sender": agent,
    }


# Combine the revision chain with the compiled graph
revision_chain = enter_chain_revision | graph_revision


# MAIN APPLICATION GRAPH

graph = StateGraph(GraphState)

# Add nodes to the main graph
graph.add_node("ask_question", ask_question)
graph.add_node("get_initial_response", get_info_for_initial_response | initial_response_chain | join_graph)
graph.add_node("get_revision_response", get_info_for_revision_response | revision_chain | join_graph)
graph.add_node("difficulty_assessment", difficulty_agent)
graph.add_node("commenter", commenter_agent)
graph.add_node("scorer", scorer_agent)
graph.add_node("check_done", check_done_agent)
graph.add_node("final_summary", final_summary_agent)
graph.add_node("beam_search_agent", beam_search_agent)
graph.add_node("initial_response_handler", initial_response_handler)
graph.add_node("revised_response_handler", revised_response_handler)

# Define edges for the main workflow
graph.set_entry_point("ask_question")
graph.add_edge("ask_question", "get_initial_response")
graph.add_edge("get_initial_response", "commenter")
graph.add_edge("commenter", "scorer")

# Define conditional edges for scoring and handling responses
graph.add_conditional_edges(
    "scorer",
    scorer_router,
    {"initial_response_handler": "initial_response_handler", "revised_response_handler": "revised_response_handler"},
)
graph.add_conditional_edges(
    "initial_response_handler",
    initial_response_router,
    {"get_initial_response": "get_initial_response", "difficulty_assessment": "difficulty_assessment", "beam_search_agent": "beam_search_agent"},
)
graph.add_conditional_edges(
    "difficulty_assessment",
    difficulty_router,
    {"get_initial_response": "get_initial_response", "beam_search_agent": "beam_search_agent"},
)

# Handle the beam search and revision edges
graph.add_edge("beam_search_agent", "get_revision_response")
graph.add_edge("get_revision_response", "commenter")

# Add conditional edges for revisions and checking completion
graph.add_conditional_edges(
    "revised_response_handler",
    revision_router,
    {"get_revision_response": "get_revision_response", "check_done": "check_done", "summary": "final_summary"},
)
graph.add_conditional_edges(
    "check_done",
    done_router,
    {"continue": "beam_search_agent", "__end__": "final_summary"},
)

# Final edge to end the process
graph.add_edge("final_summary", END)

# Compile the main graph
app = graph.compile()


# UTILITIES

def edit_prompt(agent_name):
    """
    Allows interactive editing of agent prompts using prompt_toolkit.

    Args:
        agent_name (str): The name of the agent to edit the prompt for.
    """
    agent_prompts = {
        'commenter': 'Commenter Agent',
        'difficulty': 'Difficulty Assessment Agent',
        'response': 'Initial Response Agent',
        'reviser': 'Revision Agent',
        'scorer': 'Answer Scorer Agent',
        'summary': 'Summary Agent',
        'final_summary': 'Final Summary Agent',
        'check_done': 'Check Done Agent',
    }

    key = agent_name.lower()
    if key not in agent_prompts:
        print(f"Agent '{agent_name}' not found. Available agents are:")
        for name in agent_prompts:
            print(f"- {name}")
        return

    filename = f'prompts/{key}.txt'

    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = ''

    print(f'\nEditing "{filename}". Press Esc followed by Enter to save changes.\n')

    kb = KeyBindings()

    @kb.add('c-c')
    def _(event):
        """Exit the editor without saving on Ctrl+C."""
        event.app.exit(exception=KeyboardInterrupt)

    edited_content = prompt('',
                            default=content,
                            multiline=True,
                            key_bindings=kb,
                            wrap_lines=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(edited_content)

    print(f'\nPrompt "{filename}" has been updated.')


def handle_event_logging(event_dict, log_file):
    """
    Logs event information in detail to a log file.

    Args:
        event_dict (dict): The current event data.
        log_file (file object): The open log file to write to.
    """
    if 'initial_response_handler' in event_dict:
        log_file.write("=== Initial Response ===\n")
        responses = event_dict['initial_response_handler']['responses']
        response_to_write = responses[-1]
        agent_name = response_to_write['agent_name']
        content = response_to_write['content']
        for c in content[-1:]:
            text = c.get('text', '')
            comments = c.get('comments', '')
            score = c.get('score', '')
            log_file.write(f"Agent: {agent_name}\n")
            log_file.write("Text:\n------------\n")
            log_file.write(f"{text}\n------------\n")
            log_file.write(f"Comments:\n{comments}\n")
            log_file.write(f"Score: {score}\n\n")
    elif 'revised_response_handler' in event_dict:
        index = event_dict['revised_response_handler']['index']
        responses = event_dict['revised_response_handler']['responses']
        log_file.write("=== Revised Response ===\n")
        response_to_write = responses[index - 1]
        agent_name = response_to_write['agent_name']
        content = response_to_write['content']
        for c in content[-1:]:
            text = c.get('text', '')
            comments = c.get('comments', '')
            score = c.get('score', '')
            log_file.write(f"Agent: {agent_name}\n")
            log_file.write("Text:\n------------\n")
            log_file.write(f"{text}\n------------\n")
            log_file.write(f"Comments:\n{comments}\n")
            log_file.write(f"Score: {score}\n\n")
    elif 'difficulty_assessment' in event_dict:
        difficulty = event_dict['difficulty_assessment']
        log_file.write("=== Difficulty Assessment ===\n")
        for key, value in difficulty.items():
            if key != 'start':
                log_file.write(f"{key.capitalize()}: {value}\n")
        log_file.write("\n")
    elif 'beam_search_agent' in event_dict:
        discarded_responses = event_dict['beam_search_agent'].get('discarded_responses', [])
        log_file.write("=== Discarded Responses ===\n")
        for response in discarded_responses:
            agent_name = response['agent_name']
            content = response['content']
            score = content[-1].get('score', '')
            log_file.write(f"Agent: {agent_name}\nScore: {score}\n\n")
    elif 'check_done' in event_dict:
        done = event_dict['check_done'].get('done', False)
        log_file.write("=== Check Done Result ===\n")
        log_file.write(f"Done: {done}\n\n")
    elif 'final_summary' in event_dict:
        final_response = event_dict['final_summary']['final_response']
        log_file.write("=== Final Answer ===\n")
        log_file.write(f"{final_response}\n\n")

    log_file.flush()


def main():
    """
    Main function that handles the command-line interface and runs the agents' workflows.
    """
    print("Welcome to the Test-Time Compute Simulation Tool!")
    print("\nAvailable commands:")
    print(f"{'/ask {question}':<20} {'Ask a question'}")
    print(f"{'/edit {agent_name}':<20} {'Edit an agent prompt'}")
    print(f"{'/quit':<20} {'Exit the program'}")

    while True:
        user_input = input("\nEnter a command: ").strip()

        if user_input.lower() == '/quit':
            print("Goodbye!")
            break

        elif user_input.lower().startswith('/ask'):
            parts = user_input.split(' ', 1)
            if len(parts) < 2 or not parts[1].strip():
                print("Please enter a question after '/ask'.")
                continue
            question = parts[1].strip()

            app = graph.compile()

            initial_state = {"question": question}

            log_file_folder = 'logs'
            log_file_path = f'reasoning_log_{int(time.time())}.txt'

            if not os.path.exists(log_file_folder):
                os.makedirs(log_file_folder)
                
            with open(os.path.join(log_file_folder, log_file_path), 'w', encoding='utf-8') as log_file:
                events = app.stream(initial_state, {"recursion_limit": 1000})

                for event in events:
                    event_dict = dict(event)
                    if 'ask_question' in event_dict:
                        print("Generating initial responses...")
                    elif 'initial_response_handler' in event_dict:
                        agent_name = event_dict['initial_response_handler']['responses'][-1]['agent_name']
                        print(f"Created initial {agent_name} response")
                    elif 'difficulty_assessment' in event_dict:
                        difficulty = event_dict['difficulty_assessment']['difficulty']
                        print("Assessed difficulty:", difficulty)
                    elif 'revised_response_handler' in event_dict:
                        index = event_dict['revised_response_handler']['index']
                        agent_name = event_dict['revised_response_handler']['responses'][index - 1]['agent_name']
                        print(f"Revised {agent_name} response")
                    elif 'beam_search_agent' in event_dict:
                        print("Pruning responses")
                    elif 'check_done' in event_dict:
                        done = event_dict['check_done']['done']
                        print("Decided to be done" if done else "Decided to continue")
                    elif 'final_summary' in event_dict:
                        final_response = event_dict['final_summary']['final_response']
                        print("\nFinal Answer:\n", final_response)

                    handle_event_logging(event_dict, log_file)

        elif user_input.lower().startswith('/edit'):
            parts = user_input.split(' ', 1)
            if len(parts) < 2 or not parts[1].strip():
                print("Please enter the name of an agent to edit after '/edit'.")
                continue
            agent_name = parts[1].strip()
            edit_prompt(agent_name)

        else:
            print("Unknown command. Please try again.")


if __name__ == "__main__":
    main()