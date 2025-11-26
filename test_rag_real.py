"""
Direct test of RAG engine with real Qdrant and real embeddings
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend_api.app.engine import RAGEngine
from backend_api.app.config import Settings

# Create settings with real Qdrant and real embeddings
settings = Settings(
    qdrant_url="http://localhost:6333",
    qdrant_collection="atlas_knowledge",
    embedding_model="BAAI/bge-m3",
    embedding_dimension=1024,
    similarity_threshold=0.40,  # BGE-M3 scores typically 0.4-0.6
    top_k_results=5,
    vllm_url="http://localhost:8000/v1",
    temperature=0.0
)

def test_query(question: str):
    """Test a query directly."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    
    # Create engine with real Qdrant and Embedder
    # use_mock=False means real vector store and embedder
    # But we need to manually handle LLM since vLLM is not running
    engine = RAGEngine(settings=settings, use_mock=False)
    
    # Patch the _generate method to use mock responses
    original_generate = engine._generate
    def mock_generate(prompt):
        return "Based on the retrieved documents, here is the answer. (Mock LLM response)"
    engine._generate = mock_generate
    
    try:
        response = engine.query(question)
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nConfidence: {response.confidence:.2f}")
        print(f"\nCitations ({len(response.citations)}):")
        for i, citation in enumerate(response.citations, 1):
            print(f"  {i}. {citation.title}")
            print(f"     URL: {citation.url}")
            print(f"     Score: {citation.score:.2f}")
            print(f"     Snippet: {citation.text_snippet[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing Atlas-RAG Engine with real Qdrant and embeddings\n")
    
    # Test queries about the law
    test_query("Quelles sont les conditions pour obtenir une autorisation d'exercice d'assurance?")
    test_query("Article 5")  # Test specific article retrieval
    
    # Test conversational router
    test_query("Bonjour")
    test_query("Hello")
    
    # Test queries about the portal guide
    test_query("Comment récupérer mon mot de passe oublié sur le portail?")
    test_query("Comment soumettre un formulaire sur le portail ACAPS?")
    
    print("\n" + "="*60)
    print("Testing complete!")

