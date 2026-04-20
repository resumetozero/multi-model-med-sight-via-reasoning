import os
from langchain_community.tools.tavily_search import TavilyAnswerUpload
from langchain_community.utilities import TavilySearchAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Setup Tools (Qdrant + Web Search)
search = TavilySearchAPIWrapper()
def get_latest_research(query: str):
    """Searches for the latest 2025-2026 medical research and clinical guidelines."""
    return search.run(f"latest clinical research and findings for {query} 2026")

# 2. Define the Safety & Research Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior Medical Research Assistant. 
    Your goal is to provide descriptive, evidence-based, and SAFE answers.
    
    GUIDELINES:
    1. Use the 'Local Context' (from our database) to see real patient cases.
    2. Use the 'Web Research' tool to find the latest 2026 discoveries.
    3. SAFETY: Always include a medical disclaimer. If findings are ambiguous, state so.
    4. Integration: Combine the visual evidence from our data with the theoretical research from the web."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 3. Initialize the Multi-Model Agent
llm = ChatOpenAI(model = "gpt-4-turbo-preview", temperature=0) # Or Gemini/Claude
# Add your local Qdrant retrieval tool here as well