"""
Direct test of Qdrant search
"""
from data_ingestion.vector_store import VectorStore
from data_ingestion.embedder import EmbeddingGenerator

# Connect to Qdrant
vector_store = VectorStore(
    url="http://localhost:6333",
    collection_name="atlas_knowledge",
    embedding_dimension=1024
)

# Get embedder
embedder = EmbeddingGenerator(model_name="BAAI/bge-m3")

# Test query
query = "autorisation assurance"
print(f"Query: {query}")

# Generate query embedding
query_embedding = embedder.embed_query(query)
print(f"Query embedding dimension: {len(query_embedding)}")

# Search
results = vector_store.search(
    query_embedding=query_embedding,
    top_k=5,
    score_threshold=0.3  # Very low threshold for testing
)

print(f"\nFound {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result.score:.4f}")
    print(f"   Title: {result.header_path}")
    print(f"   File: {result.file_name}")
    print(f"   Text: {result.text[:150]}...")

