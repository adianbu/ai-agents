from langgraph.prebuilt import create_react_agent
from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatOllama
# Alternative: use OpenAI-compatible API client
from langchain_openai import ChatOpenAI

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Option 1: Using OpenAI-compatible API (recommended for ML Studio)
# Assumes your ML Studio instance exposes an OpenAI-compatible endpoint
model = ChatOpenAI(
    base_url="http://localhost:1234/v1/",  # Replace with your ML Studio endpoint
    api_key="lm-studio",      # ML Studio might not require this
    model="qwen2.5-coder-32b-instruct",               # Use the actual model name in ML Studio
    temperature=0.1,
)

# Option 2: Using Ollama (if you're running Qwen through Ollama)
# model = ChatOllama(
#     base_url="http://localhost:11434",
#     model="qwen2.5-coder:32b",
#     temperature=0.1,
# )

# Option 3: Using LlamaCpp for local GGUF models
# model = LlamaCpp(
#     model_path="/path/to/qwen-coder-32b.gguf",  # Path to your GGUF model
#     temperature=0.1,
#     max_tokens=2000,
#     n_ctx=4096,  # Context window
#     verbose=False,
# )

agent = create_react_agent(
    model=model,
    tools=[get_weather],  
    prompt="You are a helpful assistant with access to tools."
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(result)