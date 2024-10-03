from typing import Tuple
import functools
import copy

from state import GraphState
from models import config, llm_mapping

# AGENTS

def create_commenter_agent(state, llm):
    """
    Creates an agent that comments on the quality of a reasoning chain.

    Args:
        state (dict): The current state containing the question and agent response.
        llm: The language model used for generating comments.

    Returns:
        dict: Updated state with comments added to the agent response.
    """
    question = state["question"]
    agent_response = state["agent_response"]
    reasoning_chain = agent_response["text"]

    # Load the system prompt for the commenter agent
    with open('prompts/commenter.txt', 'r') as file:
        system_prompt = file.read()

    # Create the prompt for the language model
    prompt = [
        ("system", system_prompt),
        ("human", f"Here is the question: {question} \nHere is the reasoning chain: {reasoning_chain} \n")
    ]

    # Generate comments using the language model
    result = llm.invoke(prompt)
    comments = result.content

    # Add comments to the agent response
    agent_response["comments"] = comments

    return {"agent_response": agent_response}


def create_scorer_agent(state, llm):
    """
    Creates an agent that scores the quality of a reasoning chain.

    Args:
        state (dict): The current state containing the question, agent response, and comments.
        llm: The language model used for generating the score.

    Returns:
        dict: Updated state with the score added to the agent response.
    """
    question = state["question"]
    agent_response = state["agent_response"]
    reasoning_chain = agent_response["text"]
    comments = agent_response["comments"]

    # Load the system prompt for the scorer agent
    with open('prompts/scorer.txt', 'r') as file:
        system_prompt = file.read()

    # Create the prompt for the language model
    prompt = [
        ("system", system_prompt),
        ("human", f"Here is the question: {question} \nHere is the reasoning chain: {reasoning_chain} \nHere are the comments: {comments} \n")
    ]

    # Generate the score using the language model
    result = llm.invoke(prompt)
    score = result.content

    # Add the score to the agent response
    agent_response["score"] = float(score)

    return {"agent_response": agent_response}


def create_difficulty_agent(state, llm):
    """
    Creates an agent that assesses the difficulty of a question.

    Args:
        state (dict): The current state containing the question, responses, comments, and grades.
        llm: The language model used for assessing difficulty.

    Returns:
        dict: A dictionary containing the difficulty level and associated parameters.
    """
    question = state["question"]
    responses_full = state["responses"]
    
    # Extract responses, comments, and grades
    responses = [response["content"][0]["text"] for response in responses_full]
    comments = [response["content"][0]["comments"] for response in responses_full]
    grades = [response["content"][0]["score"] for response in responses_full]

    # Load the system prompt for the difficulty agent
    with open('prompts/difficulty.txt', 'r') as file:
        system_prompt = file.read()

    # Create the prompt for the language model
    prompt = [
        ("system", system_prompt),
        ("human", f"Here is the question: {question} \n"
                 f"Here is the first response, comments on the response, and its grade: {responses[0]}, {comments[0]}, {grades[0]} \n"
                 f"Here is the second response, comments on the response, and its grade: {responses[1]}, {comments[1]}, {grades[1]} \n"
                 f"Here is the third response, comments on the response, and its grade: {responses[2]}, {comments[2]}, {grades[2]} \n")
    ]

    # Get the difficulty level using the language model
    result = llm.invoke(prompt)
    difficulty = int(result.content)

    # Load difficulty settings from configuration
    difficulty_settings = config['difficulty_settings']
    
    # Fetch the appropriate settings for the determined difficulty level
    settings = difficulty_settings.get(difficulty)
    if not settings:
        raise ValueError(f"Unknown difficulty level: {difficulty}")

    # Return the difficulty level and associated parameters
    return {
        "difficulty": difficulty,
        "threads": settings['threads'],
        "beams": settings['beams'],
        "start": False,
        "revisions": settings['revisions']
    }


def create_check_done_agent(state, llm):
    """
    Creates an agent that checks if the reasoning process has converged on a correct answer.

    Args:
        state (dict): The current state containing the question and responses.
        llm: The language model used for checking if the process is done.

    Returns:
        dict: A dictionary indicating whether the process is done.
    """
    question = state["question"]
    responses = [response["content"] for response in state["responses"]]

    # Load the system prompt for the check_done agent
    with open('prompts/check_done.txt', 'r') as file:
        system_prompt = file.read()

    # Create the prompt for the language model
    prompt = [
        ("system", system_prompt),
        ("human", f"Here is the question: {question} \nHere are the reasoning chains: {str(responses)} \n")
    ]

    # Invoke the model to check if the process is done
    result = llm.invoke(prompt)

    return {"done": result.content == "PROCESS DONE"}


def create_final_summary_agent(state, llm):
    """
    Creates an agent that generates a final summary of reasoning chains.

    Args:
        state (dict): The current state containing the question and responses.
        llm: The language model used for generating the final summary.

    Returns:
        dict: A dictionary containing the final combined response.
    """
    question = state["question"]
    responses = [response["content"] for response in state["responses"]]

    # Load the system prompt for the final_summary agent
    with open('prompts/final_summary.txt', 'r') as file:
        system_prompt = file.read()

    # Create the prompt for the language model
    prompt = [
        ("system", system_prompt),
        ("human", f"Here is the question: {question} \nHere are the reasoning chains: {str(responses)} \n")
    ]

    # Generate the final summary using the language model
    result = llm.invoke(prompt)

    return {"final_response": result.content}


# PARTIAL AGENT CREATION

# Create partials for each agent by binding them to their respective language model
commenter_llm = llm_mapping['commenter_agent']
commenter_agent = functools.partial(create_commenter_agent, llm=commenter_llm)

scorer_llm = llm_mapping['scorer_agent']
scorer_agent = functools.partial(create_scorer_agent, llm=scorer_llm)

difficulty_llm = llm_mapping['difficulty_agent']
difficulty_agent = functools.partial(create_difficulty_agent, llm=difficulty_llm)

check_done_llm = llm_mapping['check_done_agent']
check_done_agent = functools.partial(create_check_done_agent, llm=check_done_llm)

final_summary_llm = llm_mapping['final_summary_agent']
final_summary_agent = functools.partial(create_final_summary_agent, llm=final_summary_llm)


# STATE AND RESPONSE HANDLING FUNCTIONS

def ask_question(state: GraphState) -> GraphState:
    """
    Initialize the state with the question and default values.

    Args:
        state (GraphState): The current state.

    Returns:
        GraphState: The initial state with default values.
    """
    return {
        "initial_response_agent": config['start_settings']['starting_agent'],
        "responses": [],
        "threads": config['start_settings']['threads'],
        "start": True
    }


def get_info_for_initial_response(state: GraphState) -> Tuple[str, str]:
    """
    Get information for generating the initial response.

    Args:
        state (GraphState): The current state.

    Returns:
        Tuple[str, str]: The question and the initial response agent.
    """
    return state["question"], state["initial_response_agent"]


def get_info_for_revision_response(state: GraphState) -> Tuple[str, str, str, str]:
    """
    Get information for generating a revised response.

    Args:
        state (GraphState): The current state.

    Returns:
        Tuple[str, str, str, str]: The question, agent name, current response text, and comments.
    """
    responses = state["responses"]
    index = state["index"]
    cur_response = responses[index]
    
    return (
        state["question"], 
        cur_response["agent_name"], 
        cur_response["content"][-1]["text"], 
        cur_response["content"][-1]["comments"]
    )


def join_graph(response: dict):
    """
    Add the original response to the upper-level graph.

    Args:
        response (dict): The response to add.

    Returns:
        dict: The agent response containing the final answer.
    """
    return {"agent_response": {"text": response["final_answer"]}}


def beam_search_agent(state: GraphState) -> GraphState:
    """
    Simulates selecting the best responses using beam search.

    Args:
        state (GraphState): The current state.

    Returns:
        GraphState: The updated state with the best responses selected.
    """
    responses = state["responses"]
    responses.sort(key=lambda x: x["content"][-1]["score"], reverse=True)

    beams = state["beams"]
    threads = state["threads"]
    best_responses = responses[:beams]
    discarded_responses = responses[beams:]

    # Replicate best responses to fill threads
    final_responses = []
    for response in best_responses:
        final_responses.extend([copy.deepcopy(response)] * (threads // beams))

    # Add remaining responses to balance
    for i in range(threads % beams):
        final_responses.append(copy.deepcopy(best_responses[i]))

    return {
        "responses": final_responses, 
        "discarded_responses": discarded_responses, 
        "index": 0
    }


def initial_response_handler(state: GraphState) -> GraphState:
    """
    Handle the initial response and update the state.

    Args:
        state (GraphState): The current state.

    Returns:
        GraphState: The updated state with the initial response added.
    """
    agent_response = state["agent_response"]
    agent_name = state["initial_response_agent"]
    responses = state["responses"]

    responses.append({
        "agent_name": agent_name, 
        "content": [agent_response]
    })

    # Corrected agent_name_dict creation
    agent_names = list(llm_mapping["response_agents"].keys())  # Extract the keys (agent names) as a list

    # Create a dictionary that maps each agent name to the next one in a circular manner
    agent_name_dict = {name: agent_names[(i + 1) % len(agent_names)] for i, name in enumerate(agent_names)}

    return {"responses": responses, "initial_response_agent": agent_name_dict[agent_name]}


def revised_response_handler(state: GraphState) -> GraphState:
    """
    Handle the revised response and update the state.

    Args:
        state (GraphState): The current state.

    Returns:
        GraphState: The updated state with the revised response added.
    """
    agent_response = state["agent_response"]
    index = state["index"]
    responses = state["responses"]

    responses[index]["content"].append(agent_response)

    return {"responses": responses, "index": index + 1}