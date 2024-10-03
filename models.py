from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
import os
import yaml
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Retrieve the API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Load configuration from config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

llm_mapping = {"response_agents": {}}

def determine_llm(name, model_name):
    if name.lower().startswith('gpt'):
        return ChatOpenAI(api_key=openai_api_key, model=model_name)
    elif name.lower().startswith('claude'):
        return ChatAnthropic(api_key=anthropic_api_key, model=model_name)
    elif name.lower().startswith('mistral'):
        return ChatMistralAI(api_key=mistral_api_key, model=model_name)
    else:
        raise ValueError(f"Unknown model {model_name} for agent {name}")

# Initialize LLMs for response agents
for agent in config['llms']['response_agents']:
    name = agent['name']
    model_name = agent['model']
    llm = determine_llm(name, model_name)
    llm_mapping["response_agents"][f'{name}'] = llm


# Initialize LLM for difficulty agent
difficulty_agent = config['llms']['difficulty_agent']
name = difficulty_agent['name']
model_name = difficulty_agent['model']
llm_mapping['difficulty_agent'] = determine_llm(name, model_name)

# Initialize LLM for commenter agent
commenter_agent = config['llms']['commenter_agent']
name = commenter_agent['name']
model_name = commenter_agent['model']
llm_mapping['commenter_agent'] = determine_llm(name, model_name)

# Initialize LLM for scorer agent
scorer_agent = config['llms']['scorer_agent']
name = scorer_agent['name']
model_name = scorer_agent['model']
llm_mapping['scorer_agent'] = determine_llm(name, model_name)

# Initialize LLM for check done agent
check_done_agent = config['llms']['check_done_agent']
name = check_done_agent['name']
model_name = check_done_agent['model']
llm_mapping['check_done_agent'] = determine_llm(name, model_name)

# Initialize LLM for answer summary agent
answer_summary_agent = config['llms']['answer_summary_agent']
name = answer_summary_agent['name']
model_name = answer_summary_agent['model']
llm_mapping['answer_summary_agent'] = determine_llm(name, model_name)

# Initialize LLM for final summary agent
final_summary_agent = config['llms']['final_summary_agent']
name = final_summary_agent['name']
model_name = final_summary_agent['model']
llm_mapping['final_summary_agent'] = determine_llm(name, model_name)