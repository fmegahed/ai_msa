"""
This file is used to generate chat completions for the AI Reliability project.

Things that will need to be changed in the file:  
  - Change the `model` variable, in the models section, to the model you want to use. 
  - Remove the `head()` from the task data loading line to load all the tasks and 
    uncomment the following two lines.  

Also, note that I have now embedded the rubric in the system prompt. I was getting an error
(when it was a dictionary like our previous approach). 

In general, I think it would probably be easier to make a copy of this file and run
it for each example in our test cases. This means that:  
  - We will likely want to have seperate data files for each test case.  
  - System prompt and user prompt will change for each test case.
"""

# ------------------------------------------------------------------------------
# Import Libraries
# ----------------

# Standard Libraries
import os

# External Libraries
import pandas as pd
import datetime as dt

from dotenv import load_dotenv
from itertools import product

from langchain.prompts.chat import ChatPromptTemplate
# see https://python.langchain.com/docs/modules/model_io/chat/quick_start/
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq



# ------------------------------------------------------------------------------
# Models:
# -------

# This list includes models from various platforms, categorized by the platform they originate from:
# - ChatGPT models are detailed in the OpenAI documentation: https://platform.openai.com/docs/models
# - Anthropic models are documented here: https://docs.anthropic.com/claude/docs/models-overview#model-comparison
# - Cohere model documentation can be found at: https://docs.cohere.com/
# - Google AI Models are documented in: # https://ai.google.dev/docs
# - Mistral AI models are documented in https://docs.mistral.ai/guides/model-selection/
# Models explored based on documentation from https://python.langchain.com/docs/modules/model_io/
models = [
  'gpt-4-turbo-preview', 'gpt-3.5-turbo', 
  'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307',
  'command-r-plus',
  'gemini-pro',
  'open-mistral-7b', 'mistral-medium-latest',
  'llama3-8b-8192', 'llama3-70b-8192',
  'gemma-7b-it'
  ]

model = models[10] # change this to the model you want to use



# ------------------------------------------------------------------------------
# Environment Variables:
# ----------------------

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
mistral_api_key = os.getenv('MISTRAL_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')




# ------------------------------------------------------------------------------
# Task Data and Rubric:
# ----------------------

# Task Data:
# ----------

num_tasks = 3 # for the task_statements.csv and not for the sample since it contains 1 row

task = (
  pd.read_csv('data/sample_task_statements.csv')
  # .sample(num_tasks, random_state=2024) # randomly sample num_rows with a fixed seed corresponding to the year 2024
  # .reset_index(drop=True)
  )



# ------------------------------------------------------------------------------
# System, User and Final Chat Prompt:
# -----------------------------------

system_prompt = f"""
Consider the most powerful OpenAI large language model (LLM). This model can complete 
many tasks that can be formulated as having text input and text output where the context 
for the input can be captured in 2000 words. The model also cannot draw up-to-date facts 
(those from <1 year ago) unless they are captured in the input. Assume you are a worker 
with an average level of expertise in your role trying to complete the given task. You 
have access to the LLM as well as any other existing software or computer hardware tools 
mentioned in the task. You also have access to any commonly available technical tools 
accessible via a laptop (e.g. a microphone, speakers, etc.). You do not have access to 
any other physical tools or materials. You are a helpful research assistant who wants to 
label the given tasks according to the rubric below. Equivalent quality means someone 
reviewing the work would not be able to tell whether a human completed it on their own 
or with assistance from the LLM. If you aren’t sure how to judge the amount of time a 
task takes, consider whether the tools described exposed the majority of subtasks 
associated with the task.

Rubric:
  * E1: Label tasks E1 if direct access to the LLM through an interface like ChatGPT or the OpenAI playground alone can reduce the time it takes to complete the task with equivalent quality by at least half. This includes tasks that can be reduced to: - Writing and transforming text and code according to complex instructions, - Providing edits to existing text or code following specifications, - Writing code that can help perform a task that used to be done by hand, - Translating text between languages, - Summarizing medium-length documents, - Providing feedback on documents, - Answering questions about a document, - Generating questions a user might want to ask about a document, - Writing questions for an interview or assessment, - Writing and responding to emails, including ones that involve refuting information or engaging in a negotiation (but only if the negotiation is via written correspondence), - Maintain records of written data, - Prepare training materials based on general knowledge, or - Inform anyone of any information via any written or spoken medium.  
  * E2: Label tasks E2 if having access to the LLM alone may not reduce the time it takes to complete the task by at least half, but it is easy to imagine additional software that could be developed on top of the LLM that would reduce the time it takes to complete the task by half. This software may include capabilities such as: - Summarizing documents longer than 2000 words and answering questions about those documents, - Retrieving up-to-date facts from the Internet and using those facts in combination with the LLM capabilities, - Searching over an organization’s existing knowledge, data, or documents and retrieving information, - Retrieving highly specialized domain knowledge, - Make recommendations given data or written input, - Analyze written information to inform decisions, - Prepare training materials based on highly specialized knowledge, - Provide counsel on issues, and - Maintain complex databases.  
  * E3: Suppose you had access to both the LLM and a system that could view, caption, and create images as well as any systems powered by the LLM (those in E2 above). This system cannot take video as an input and it cannot produce video as an output. This system cannot accurately retrieve very detailed information from image inputs, such as measurements of dimensions within an image. Label tasks as E3 if there is a significant reduction in the time it takes to complete the task given access to a LLM and these image capabilities: - Reading text from PDFs, - Scanning images, or - Creating or editing digital images according to instructions. The images can be realistic but they should not be detailed. The model can identify objects in the image but not relationships between those options.  
  * E0: Label tasks E0 if none of the above clearly decrease the time it takes for an experienced worker to complete the task with high quality by at least half. Some examples: - If a task requires a high degree of human interaction.  


Your role:
You will be provided with an occupation title and a task description. Then, you must do 
three things. 
1: Reason step by step to decide which of the labels (E0/E1/E2/E3) from the exposure 
rubric you were given applies to the task’s exposure to LLM. Report the label. Give an 
explanation for which label applies and report the label that you think fits best. 
Do not say zero or N/A.
2: Report only the label that you determined for the task, which should match the 
label in step 1.  Do not reply N/A.  
3: Given the amount of speculation required in step 1, describe your certainty about the 
estimate–either high, moderate, or low. Ensure that your response is consistent with the 
provided descriptions, seperating each of the three things with two new lines. Each of 
the things must start with its number followed by a colon and a space. For example, 
'1: This is the first thing.' Do not return * as part of your response. Stick to the 
guidelines as the response will be parsed into three seperate columns.

"""

user_prompt = "Task: {task_description}\nOccupation: {occupation}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", user_prompt),
])



# ------------------------------------------------------------------------------
# Chat Completion Function:
# -------------------------

def generate_chat_completion(data, index, temp=0, max_num_tokens = 500, save_to_csv = True, output_file='results/ai_exposure_completions.csv'):
  """
  Function to generate chat completions, with the data, model and index as inputs.
  We also have the option to save the output to a csv file, and include the output file name as an input.
  We provide reasonable defaults for traditional chat completions parameters.
  """
  messages = chat_prompt.format_messages(
    occupation = data.loc[index, 'occupation'],
    task_description = data.loc[index, 'task_description']
    )
  
  # extracting the model from the data
  model = data.loc[index, 'model']
  
  if model == 'gpt-4-turbo-preview': 
    chat_model = ChatOpenAI(model="gpt-4-turbo-preview", temperature=temp, max_tokens=max_num_tokens)  
  elif model == 'gpt-3.5-turbo':
    chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=temp, max_tokens=max_num_tokens)
  elif model == "claude-3-opus-20240229": 
    chat_model = ChatAnthropic(model="claude-3-opus-20240229", temperature=temp, max_tokens=max_num_tokens)
  elif model == "claude-3-sonnet-20240229":
    chat_model = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=temp, max_tokens=max_num_tokens)
  elif model == "claude-3-haiku-20240307":
    chat_model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=temp, max_tokens=max_num_tokens)
  elif model == "command-r-plus":
    chat_model = ChatCohere(model="command-r-plus", temperature=temp, max_tokens=max_num_tokens)
  elif model == "gemini-pro": 
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True, temperature=temp, max_tokens=max_num_tokens)
  elif model == "open-mistral-7b": 
    chat_model = ChatMistralAI(model="open-mistral-7b", temperature=temp, max_tokens=max_num_tokens)
  elif model == "mistral-medium-latest":
    chat_model = ChatMistralAI(model="mistral-medium-latest", temperature=temp, max_tokens=max_num_tokens)
  elif model == "llama3-8b-8192":
    chat_model = ChatGroq(model="llama3-8b-8192", temperature=temp, max_tokens=max_num_tokens)
  elif model == "llama3-70b-8192":
    chat_model = ChatGroq(model="llama3-70b-8192", temperature=temp, max_tokens=max_num_tokens)
  elif model == "gemma-7b-it":
    chat_model = ChatGroq(model="gemma-7b-it", temperature=temp, max_tokens=max_num_tokens)
  else:
      raise ValueError(f"Model {model} is not supported.")  

  # generating the response and extracting the content
  chat_response = chat_model.invoke(messages)
  chat_response_content = chat_response.content
  chat_response_id = chat_response.id
  
  data_row = pd.DataFrame({
    'date': [dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
    'model': [model],
    'occupation': [data.loc[index, 'occupation']],
    'task_id': [data.loc[index, 'Task ID']],
    'task_description': [data.loc[index, 'task_description']],
    'chat_response': [chat_response_content], 
    'chat_response_id': [chat_response_id]
    })
  
  # check if we do not want to save to csv
  if not save_to_csv:
    return chat_response_content
  # if we want to save to csv and the file does not exist
  elif not os.path.exists(output_file):
    df = data_row
    df.to_csv(output_file, index=False)
  # otherwise, we append to the existing file
  else:
    df = pd.read_csv(output_file)
    df = pd.concat([df, data_row], ignore_index=True)
    df.to_csv(output_file, index=False)
  return chat_response_content



# ------------------------------------------------------------------------------
# Tests for the function:
# -----------------------
sample_models = [models[11]]
replicates = 2

# Combinations of models and replicates
combinations = list(product(sample_models, range(1, replicates + 1)))

# Creating the data with replicates, while adding the model and replicate columns
# and shuffling the data
data = pd.concat([task] * len(combinations), ignore_index=True)
data['model'] = [comb[0] for comb in combinations]
data['replicate'] = [comb[1] for comb in combinations]
data = data.sample(frac=1, random_state=2024).reset_index(drop=True)

# Creating a new column replicated that will be used to keep track of the actual 
# order of replicates and removing the replicate column
data['replicated'] = data.groupby(['model', 'Task ID']).cumcount() + 1
data.drop(columns = 'replicate', inplace=True)

# Running the function for each row in the data
for index in range(data.shape[0]):
  generate_chat_completion(data, index, save_to_csv = True, output_file='results/ai_exposure_completions.csv')



# ------------------------------------------------------------------------------
