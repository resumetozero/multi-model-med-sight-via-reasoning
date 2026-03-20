from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

# 1. Connect to Qdrant (The Librarian)
q_client = QdrantClient(url="YOUR_QDRANT_URL", api_key="YOUR_KEY")
vectorstore = Qdrant(client=q_client, collection_name="medical_research", embeddings=medical_embeddings)

# 2. Define the RAG Tool
def get_medical_context(query: str):
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

# --- Updated Reflector Node ---
def call_reflector(state: AgentState):
    # Step A: Get research data about the Vision Agent's findings
    findings = state['vision_description']
    research_context = get_medical_context(findings)
    
    # Step B: The "Consultation"
    prompt = f"""
    Vision Analysis: {findings}
    Medical Research: {research_context}
    
    CRITIQUE TASK: 
    1. Does the research support the Vision findings?
    2. Is there a contradiction?
    If 100% accurate, start with 'APPROVED'. 
    If not, provide specific corrections based on the Research text.
    """
    response = reflector_model.invoke(prompt)
    return {"critique": response.content}