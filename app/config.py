from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(
    model="gpt-4o",  
    temperature=0.1,
    api_key="OPENAI_KEY"
)

# Example usage
response = llm.invoke("Explain the difference between Bedrock and OpenAI APIs")
print(response.content)
