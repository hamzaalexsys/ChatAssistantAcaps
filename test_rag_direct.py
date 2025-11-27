"""
Direct test of RAG engine with real Qdrant and mock LLM
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend_api.app.engine import RAGEngine
from backend_api.app.config import Settings

# Create settings with real Qdrant but we'll use mock LLM
settings = Settings(
    qdrant_url="http://localhost:6333",
    qdrant_collection="atlas_knowledge",
    embedding_model="BAAI/bge-m3",
    embedding_dimension=1024,
    similarity_threshold=0.75,
    top_k_results=5,
    vllm_url="http://localhost:8000/v1",  # Won't be used in mock mode
    temperature=0.0
)

def test_query(question: str):
    """Test a query directly."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    
    # Create engine with mock LLM but real Qdrant
    engine = RAGEngine(settings=settings, use_mock=True)
    
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
    print("Testing Atlas-RAG Engine directly with real Qdrant\n")
    
    # Test queries about the law
    test_query("Quelles sont les conditions pour obtenir une autorisation d'exercice d'assurance?")
    test_query("Quelles sont les sanctions en cas de violation de la loi sur les assurances?")
    
    # Test queries about the portal guide
    test_query("Comment récupérer mon mot de passe oublié sur le portail?")
    test_query("Comment soumettre un formulaire sur le portail ACAPS?")
    
    print("\n" + "="*60)
    print("Testing complete!")


