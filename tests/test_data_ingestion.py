"""
Unit Tests for Data Ingestion Pipeline
Tests parser, embedder, vector store, and pipeline integration.
"""
import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion.parser import MarkdownParser, DocumentChunk, parse_markdown
from data_ingestion.embedder import MockEmbeddingGenerator, EmbeddingResult
from data_ingestion.vector_store import MockVectorStore, SearchResult
from data_ingestion.pipeline import IngestionPipeline, IngestionResult
from data_ingestion.config import IngestionConfig


class TestMarkdownParser:
    """Tests for the Markdown parser."""
    
    @pytest.fixture
    def sample_markdown(self):
        """Sample Markdown content with frontmatter."""
        return """---
title: Test Document
base_url: https://example.com/docs
---

# Main Title

Introduction paragraph.

## Section 1

Content of section 1.

### Subsection 1.1

Content of subsection 1.1.

## Section 2

Content of section 2.
"""

    @pytest.fixture
    def temp_md_file(self, sample_markdown):
        """Create a temporary Markdown file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(sample_markdown)
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    def test_parser_initialization(self):
        """Test parser can be initialized."""
        parser = MarkdownParser()
        assert parser is not None
        assert parser.default_base_url == ""
        
        parser_with_url = MarkdownParser(default_base_url="https://test.com")
        assert parser_with_url.default_base_url == "https://test.com"
    
    def test_parse_file_extracts_frontmatter(self, temp_md_file):
        """Test that frontmatter is correctly extracted."""
        parser = MarkdownParser()
        chunks = parser.parse_file(temp_md_file)
        
        assert len(chunks) > 0
        # Check base_url is extracted from frontmatter
        assert all(c.base_url == "https://example.com/docs" for c in chunks)
    
    def test_parse_file_creates_chunks_by_headers(self, temp_md_file):
        """Test that content is split by headers."""
        parser = MarkdownParser()
        chunks = parser.parse_file(temp_md_file)
        
        # Should create chunks for each section
        assert len(chunks) >= 3
        
        # Check header paths are built correctly
        header_paths = [c.header_path for c in chunks]
        assert any("Main Title" in hp for hp in header_paths)
        assert any("Section 1" in hp for hp in header_paths)
        assert any("Section 2" in hp for hp in header_paths)
    
    def test_parse_file_generates_url_slugs(self, temp_md_file):
        """Test URL slug generation."""
        parser = MarkdownParser()
        chunks = parser.parse_file(temp_md_file)
        
        # Check slugs are generated
        for chunk in chunks:
            assert chunk.url_slug != ""
            assert "#" in chunk.url_slug  # Should contain anchor
    
    def test_parse_file_preserves_hierarchy(self, temp_md_file):
        """Test hierarchical header paths."""
        parser = MarkdownParser()
        chunks = parser.parse_file(temp_md_file)
        
        # Find subsection chunk
        subsection_chunks = [c for c in chunks if "Subsection 1.1" in c.header_path]
        assert len(subsection_chunks) > 0
        
        # Should include parent headers
        assert "Section 1" in subsection_chunks[0].header_path
    
    def test_parse_file_not_found(self):
        """Test handling of non-existent file."""
        parser = MarkdownParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file("/nonexistent/path/file.md")
    
    def test_document_chunk_to_dict(self):
        """Test DocumentChunk serialization."""
        chunk = DocumentChunk(
            text="Test content",
            file_name="test.md",
            header_path="Section 1",
            url_slug="https://test.com#section-1",
            base_url="https://test.com",
            last_updated="2024-01-01T00:00:00",
            chunk_index=0,
            metadata={"custom": "value"}
        )
        
        data = chunk.to_dict()
        assert data["text"] == "Test content"
        assert data["file_name"] == "test.md"
        assert data["header_path"] == "Section 1"
        assert data["custom"] == "value"


class TestEmbeddingGenerator:
    """Tests for the embedding generator."""
    
    def test_mock_embedder_initialization(self):
        """Test mock embedder can be initialized."""
        embedder = MockEmbeddingGenerator(dimension=1024)
        assert embedder.dimension == 1024
        assert embedder.model_name == "mock-embedding-model"
    
    def test_mock_embedder_single_text(self):
        """Test embedding single text."""
        embedder = MockEmbeddingGenerator(dimension=512)
        result = embedder.embed_text("Test sentence")
        
        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 512
        assert result.text == "Test sentence"
    
    def test_mock_embedder_deterministic(self):
        """Test embeddings are deterministic for same text."""
        embedder = MockEmbeddingGenerator(dimension=256)
        
        result1 = embedder.embed_text("Hello world")
        result2 = embedder.embed_text("Hello world")
        
        assert result1.embedding == result2.embedding
    
    def test_mock_embedder_different_texts(self):
        """Test different texts produce different embeddings."""
        embedder = MockEmbeddingGenerator(dimension=256)
        
        result1 = embedder.embed_text("Hello world")
        result2 = embedder.embed_text("Goodbye world")
        
        assert result1.embedding != result2.embedding
    
    def test_mock_embedder_batch(self):
        """Test batch embedding."""
        embedder = MockEmbeddingGenerator(dimension=128)
        texts = ["Text 1", "Text 2", "Text 3"]
        
        results = embedder.embed_texts(texts)
        
        assert len(results) == 3
        assert all(len(r.embedding) == 128 for r in results)
    
    def test_mock_embedder_normalized(self):
        """Test embeddings are normalized."""
        import numpy as np
        embedder = MockEmbeddingGenerator(dimension=256)
        
        result = embedder.embed_text("Test")
        norm = np.linalg.norm(result.embedding)
        
        assert abs(norm - 1.0) < 0.001  # Should be unit normalized


class TestVectorStore:
    """Tests for the vector store."""
    
    @pytest.fixture
    def mock_store(self):
        """Create a mock vector store."""
        return MockVectorStore(collection_name="test", embedding_dimension=128)
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample document chunks."""
        return [
            DocumentChunk(
                text="Article 5 discusses sick leave policies.",
                file_name="rules.md",
                header_path="Section 2 > Article 5",
                url_slug="https://test.com#article-5",
                base_url="https://test.com",
                last_updated="2024-01-01",
                chunk_index=0
            ),
            DocumentChunk(
                text="Article 6 covers vacation days.",
                file_name="rules.md",
                header_path="Section 2 > Article 6",
                url_slug="https://test.com#article-6",
                base_url="https://test.com",
                last_updated="2024-01-01",
                chunk_index=1
            ),
        ]
    
    def test_mock_store_creation(self, mock_store):
        """Test mock store initialization."""
        assert mock_store.collection_name == "test"
        assert mock_store.embedding_dimension == 128
        assert mock_store.health_check() is True
    
    def test_mock_store_upsert(self, mock_store, sample_chunks):
        """Test upserting chunks."""
        embedder = MockEmbeddingGenerator(dimension=128)
        embeddings = [embedder.embed_text(c.text).embedding for c in sample_chunks]
        
        count = mock_store.upsert_chunks(sample_chunks, embeddings)
        
        assert count == 2
        assert len(mock_store.points) == 2
    
    def test_mock_store_search(self, mock_store, sample_chunks):
        """Test searching the store."""
        embedder = MockEmbeddingGenerator(dimension=128)
        embeddings = [embedder.embed_text(c.text).embedding for c in sample_chunks]
        mock_store.upsert_chunks(sample_chunks, embeddings)
        
        # Search for sick leave
        query_embedding = embedder.embed_query("sick leave policy")
        results = mock_store.search(query_embedding, top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
        # Cosine similarity can range from -1 to 1 for random vectors
        assert all(-1 <= r.score <= 1 for r in results)
    
    def test_mock_store_delete_by_file(self, mock_store, sample_chunks):
        """Test deleting chunks by file name."""
        embedder = MockEmbeddingGenerator(dimension=128)
        embeddings = [embedder.embed_text(c.text).embedding for c in sample_chunks]
        mock_store.upsert_chunks(sample_chunks, embeddings)
        
        deleted = mock_store.delete_by_file("rules.md")
        
        assert deleted == 2
        assert len(mock_store.points) == 0
    
    def test_mock_store_collection_info(self, mock_store, sample_chunks):
        """Test getting collection info."""
        embedder = MockEmbeddingGenerator(dimension=128)
        embeddings = [embedder.embed_text(c.text).embedding for c in sample_chunks]
        mock_store.upsert_chunks(sample_chunks, embeddings)
        
        info = mock_store.get_collection_info()
        
        assert info["name"] == "test"
        assert info["points_count"] == 2


class TestIngestionPipeline:
    """Tests for the full ingestion pipeline."""
    
    @pytest.fixture
    def sample_doc(self):
        """Create a sample document file."""
        content = """---
title: Test Rules
base_url: https://test.acaps.ma/rules
---

# Test Rules Document

## Article 1

This is the first article with important information.

## Article 2

This is the second article about different topics.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    def test_pipeline_initialization_mock(self):
        """Test pipeline initializes with mock components."""
        pipeline = IngestionPipeline(use_mock=True)
        
        assert pipeline is not None
        assert pipeline.use_mock is True
    
    def test_pipeline_run_single_file(self, sample_doc):
        """Test pipeline processes a single file."""
        pipeline = IngestionPipeline(use_mock=True)
        
        result = pipeline.run(file_path=sample_doc)
        
        assert isinstance(result, IngestionResult)
        assert result.files_processed == 1
        assert result.chunks_created > 0
        assert result.chunks_upserted > 0
        assert result.success is True
    
    def test_pipeline_result_structure(self, sample_doc):
        """Test the result structure."""
        pipeline = IngestionPipeline(use_mock=True)
        result = pipeline.run(file_path=sample_doc)
        
        assert hasattr(result, 'files_processed')
        assert hasattr(result, 'chunks_created')
        assert hasattr(result, 'chunks_upserted')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'success')
    
    def test_pipeline_status(self):
        """Test getting pipeline status."""
        pipeline = IngestionPipeline(use_mock=True)
        status = pipeline.get_status()
        
        assert "config" in status
        assert "vector_store" in status
        assert "health" in status
        assert status["health"] is True
    
    def test_pipeline_nonexistent_file(self):
        """Test pipeline handles missing file gracefully."""
        pipeline = IngestionPipeline(use_mock=True)
        result = pipeline.run(file_path="/nonexistent/file.md")
        
        assert result.success is False
        assert len(result.errors) > 0


class TestConfiguration:
    """Tests for configuration handling."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = IngestionConfig()
        
        assert config.embedding_dimension == 1024
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = IngestionConfig()
        assert config.validate() is True
        
        # Test invalid config
        config.embedding_dimension = -1
        with pytest.raises(ValueError):
            config.validate()
    
    def test_config_chunk_overlap_validation(self):
        """Test chunk overlap validation."""
        config = IngestionConfig()
        config.chunk_overlap = config.chunk_size + 1
        
        with pytest.raises(ValueError, match="Chunk overlap must be less than chunk size"):
            config.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

