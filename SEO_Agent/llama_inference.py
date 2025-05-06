# Example: reuse your existing OpenAI setup
from openai import OpenAI
from system_prompt import react_system_prompt
from SimplerLLM.tools.rapid_api import RapidAPIClient
from SimplerLLM.tools.json_helpers import extract_json_from_text
from SimplerLLM.language.llm import LLM,LLMProvider
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
# client =LLM.create(LLMProvider.OPENAI,model_name="llama-3.2-3b-instruct")
# Define function schema for the AI to understand available actions
function_schema = """
Available functions:
1. get_response_time(url: string) -> float
   Returns the response time of the given URL in seconds.
2. get_seo_page_report(url: string) -> dict
   Returns SEO analysis report for the given URL.
"""

def get_seo_page_report(url :str):
    api_url = "https://website-seo-analyzer.p.rapidapi.com/seo/seo-audit-basic"
    api_params = {
        'url': url,
    }
    api_client = RapidAPIClient() 
    response = api_client.call_api(api_url, method='GET', params=api_params)
    return response
# Update the system prompt to include available functions
system_message = f"{react_system_prompt}\n\n{function_schema}"

user_prompt = "what is the response time of github.com?"

messages = [
    {"role": "system", "content": system_message},  # Use the combined system_message
    {"role": "user", "content": user_prompt},
]

def get_response_time(url):
    if url == "github.com/adianbu":
        return 0.5
    if url == "google.com":
        return 0.3
    if url == "openai.com":
        return 0.4
    return -1  # Return -1 for unknown URLs

completion = client.chat.completions.create(
  model="llama-3.2-3b-instruct",
  messages=messages,
  temperature=0.7,
)

print(completion.choices[0].message)

# Fix the available_actions dictionary - store the function reference, not the result
available_actions = {
    "get_response_time": get_response_time,  # Store function reference
    "get_seo_page_report": get_seo_page_report  # Store function reference
}

turn_count = 1
max_turns = 5


while turn_count < max_turns:
    print (f"Loop: {turn_count}")
    print("----------------------")
    turn_count += 1

    response = completion.choices[0].message.content

    print(response)

    json_function = extract_json_from_text(response)

    if json_function:
            function_name = json_function[0]['function_name']
            function_parms = json_function[0]['function_parms']
            if function_name not in available_actions:
                raise Exception(f"Unknown action: {function_name}: {function_parms}")
            print(f" -- running {function_name} {function_parms}")
            action_function = available_actions[function_name]
            #call the function
            result = action_function(**function_parms)
            function_result_message = f"Action_Response: {result}"
            messages.append({"role": "user", "content": function_result_message})
            print(function_result_message)
    else:
         break