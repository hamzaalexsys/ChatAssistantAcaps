"""
Test script for Atlas-RAG API queries
"""
import requests
import json

API_BASE = "http://localhost:8080"

def test_query(question: str):
    """Test a query against the API."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    
    response = requests.post(
        f"{API_BASE}/chat",
        json={"question": question}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nAnswer:\n{data['answer']}")
        print(f"\nConfidence: {data['confidence']:.2f}")
        print(f"\nCitations ({len(data['citations'])}):")
        for i, citation in enumerate(data['citations'], 1):
            print(f"  {i}. {citation['title']}")
            print(f"     URL: {citation['url']}")
            print(f"     Score: {citation['score']:.2f}")
            print(f"     Snippet: {citation['snippet'][:100]}...")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("Testing Atlas-RAG API with sample queries\n")
    
    # Test queries about the law
    test_query("Quelles sont les conditions pour obtenir une autorisation d'exercice d'assurance?")
    test_query("Quelles sont les sanctions en cas de violation de la loi sur les assurances?")
    
    # Test queries about the portal guide
    test_query("Comment récupérer mon mot de passe oublié sur le portail?")
    test_query("Comment soumettre un formulaire sur le portail ACAPS?")
    
    print("\n" + "="*60)
    print("Testing complete!")

