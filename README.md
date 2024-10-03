# Test-Time Compute Simulation

Welcome to the Test-Time Compute Simulation Tool! This application allows you to interact with AI agents to answer complex questions, giving you freedom to alter the specifics of your test-time compute algorithm. This project was inspired by this research paper: https://arxiv.org/pdf/2408.03314.

## Theory
This application is a simplification of the procedure described in the paper linked above. Here is a high level overview:

- **Initial Response Generation**: first, we generate a certain amount of responses to the question.
- **Comments and Scores**: Each response generated receives comments on its accuracy and logical structure. It also receives a numeric grade out of 10 on the quality of the response.
- **Difficulty Assessment**: after the initial responses have been commented on and scored, we assess how difficult the question.
- **Best-of-N vs. Revisions vs. Beam Search**: given the difficulty, we need to distribute our compute resources. We can generate more individual threads (best-of-N), or generate fewer individual threads, and revise more. After each revision step, we can also choose to discard our worst responses (beam search). See `config.yaml` to learn more about this compute resource distribution.
- **Final Summary**: Once we decide we are done, or are out of compute, we generate a final, coherent answer to the user.


## Features

- **Complex Reasoning**: ask questions requiring complex reasoning, and watch the application reason through various responses, revising along the way to reach a final answer. Just use `/ask`! Check out a complete log of intermediary reasoning steps under the `/logs` directory.

- **Customizable Prompts**: Using the `/edit` command, edit any of the agents' system prompts - simply type the name of the agent you want to edit. For example, `/edit commenter` will edit the agent that provides comments to the responses. Here are the valid agents to edit:
    - commenter
    - difficulty
    - response
    - reviser
    - scorer
    - summary
    - final_summary
    - check_done

- **Configurable Settings**: Simply open up `config.yaml` and configure any setting you wish. More detailed instructions in the file. 

- **Full reasoning logs**: If you look in the `/logs` directory, you will see a full, detailed log of all intermediary outputs and steps. 

## Installation

### Prerequisites

- **Python**: Ensure you have Python 3.7 or higher installed. [Download Python](https://www.python.org/downloads/)

### 🔧 Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/eligotts/test-time-compute.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd test-time-compute
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your API keys**:

   Create a `.env` file in the root directory, and enter your API keys

   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Usage

Launch the application using the following command:

```bash
python o1_simulation.py
```

### Available Commands

- `/ask`: Ask a question! See full output in `/logs` directory

- `/edit`: Edit system prompts of agents. Available agents to edit are:
    - commenter
    - difficulty
    - response
    - reviser
    - scorer
    - summary
    - final_summary
    - check_done

- `/quit`: Exit the program

### Please edit and add stuff if you're interested! 