"""
Main Data Ingestion Pipeline for Atlas-Hyperion v3.0
Orchestrates parsing, contextual crystallization, graph extraction,
embedding, and upserting to Qdrant.
"""
import os
import logging
import argparse
from typing import List, Optional
from dataclasses import dataclass

from .config import get_config, IngestionConfig
from .parser import MarkdownParser, DocumentChunk
from .embedder import EmbeddingGenerator, get_embedding_generator
from .vector_store import VectorStore
from .contextualizer import get_contextualizer, SimpleContextualizer
from .graph_extractor import GraphExtractor, get_graph_extractor

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of ingestion pipeline run."""
    files_processed: int
    chunks_created: int
    chunks_upserted: int
    errors: List[str]
    # Atlas-Hyperion v3.0 stats
    chunks_crystallized: int = 0
    edges_extracted: int = 0
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class IngestionPipeline:
    """
    Main ingestion pipeline for Atlas-Hyperion v3.0.
    Processes Markdown files with contextual crystallization,
    graph extraction, embedding, and storage in Qdrant.
    """
    
    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        use_mock: bool = False,
        enable_crystallization: bool = True,
        enable_graph_extraction: bool = True,
        use_simple_contextualizer: bool = True  # Use simple mode by default (no LLM)
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            config: Configuration object (uses default if None)
            use_mock: Use mock components for testing
            enable_crystallization: Enable contextual crystallization
            enable_graph_extraction: Enable graph edge extraction
            use_simple_contextualizer: Use simple contextualizer (no LLM required)
        """
        self.config = config or get_config()
        self.use_mock = use_mock
        self.enable_crystallization = enable_crystallization
        self.enable_graph_extraction = enable_graph_extraction
        
        # Initialize components
        self.parser = MarkdownParser()
        
        # Initialize contextualizer (Atlas-Hyperion v3.0)
        self.contextualizer = get_contextualizer(
            use_mock=use_mock,
            use_simple=use_simple_contextualizer
        )
        
        # Initialize graph extractor (Atlas-Hyperion v3.0)
        self.graph_extractor = get_graph_extractor()
        
        if use_mock:
            from .embedder import MockEmbeddingGenerator
            from .vector_store import MockVectorStore
            self.embedder = MockEmbeddingGenerator(dimension=self.config.embedding_dimension)
            self.vector_store = MockVectorStore(
                collection_name=self.config.collection_name,
                embedding_dimension=self.config.embedding_dimension
            )
        else:
            self.embedder = get_embedding_generator(
                model_name=self.config.embedding_model,
                use_mock=False
            )
            self.vector_store = VectorStore(
                url=self.config.qdrant_url,
                collection_name=self.config.collection_name,
                embedding_dimension=self.config.embedding_dimension
            )
        
        logger.info(f"IngestionPipeline initialized (mock={use_mock}, crystallization={enable_crystallization}, graph={enable_graph_extraction})")
    
    def run(
        self,
        source_dir: Optional[str] = None,
        file_path: Optional[str] = None,
        recreate_collection: bool = False
    ) -> IngestionResult:
        """
        Run the ingestion pipeline.
        
        Args:
            source_dir: Directory containing Markdown files
            file_path: Single file to process (overrides source_dir)
            recreate_collection: Whether to recreate the collection
            
        Returns:
            IngestionResult with statistics
        """
        errors = []
        chunks: List[DocumentChunk] = []
        files_processed = 0
        
        # Step 1: Create/verify collection
        try:
            self.vector_store.create_collection(recreate=recreate_collection)
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return IngestionResult(0, 0, 0, [str(e)])
        
        # Step 2: Parse documents
        try:
            if file_path:
                logger.info(f"Processing single file: {file_path}")
                chunks = self.parser.parse_file(file_path)
                files_processed = 1
            else:
                source = source_dir or self.config.documents_dir
                logger.info(f"Processing directory: {source}")

                # For directory processing, check for new or updated files
                if not recreate_collection:
                    all_files = []
                    import glob
                    import os
                    from datetime import datetime
                    
                    file_pattern = os.path.join(source, "*.md")
                    for file_path in glob.glob(file_pattern):
                        file_name = os.path.basename(file_path)
                        
                        # Check if file exists in vector store
                        stored_last_updated = self.vector_store.get_file_last_updated(file_name)
                        
                        if stored_last_updated:
                            # File exists, check if it's outdated
                            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                            
                            # Simple string comparison works for ISO format
                            if file_mtime <= stored_last_updated:
                                logger.info(f"Skipping up-to-date file: {file_name}")
                                continue
                            else:
                                logger.info(f"File updated detected: {file_name} (Disk: {file_mtime} > Store: {stored_last_updated})")
                                # We need to delete old chunks before processing new ones
                                self.vector_store.delete_by_file(file_name)
                        else:
                            logger.info(f"New file detected: {file_name}")
                        
                        all_files.append(file_path)

                    # Parse only non-indexed or updated files
                    all_chunks = []
                    for file_path in all_files:
                        try:
                            file_chunks = self.parser.parse_file(file_path)
                            all_chunks.extend(file_chunks)
                        except Exception as e:
                            logger.error(f"Error parsing {file_path}: {e}")
                            continue

                    chunks = all_chunks
                else:
                    # Recreate collection - process all files
                    chunks = self.parser.parse_directory(source)

                files_processed = len(set(c.file_name for c in chunks))
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            errors.append(f"Parsing error: {e}")
            return IngestionResult(files_processed, 0, 0, errors)
        
        if not chunks:
            logger.warning("No chunks created from documents")
            return IngestionResult(files_processed, 0, 0, ["No chunks created"])
        
        logger.info(f"Created {len(chunks)} chunks from {files_processed} files")
        
        # Step 3: Contextual Crystallization (Atlas-Hyperion v3.0)
        chunks_crystallized = 0
        if self.enable_crystallization:
            try:
                logger.info("Crystallizing chunks with contextual enrichment...")
                # Group chunks by file for document-level context
                chunks_by_file = {}
                for chunk in chunks:
                    if chunk.file_name not in chunks_by_file:
                        chunks_by_file[chunk.file_name] = []
                    chunks_by_file[chunk.file_name].append(chunk)
                
                for file_name, file_chunks in chunks_by_file.items():
                    # Get document title from first chunk's metadata or file name
                    doc_title = file_name.replace('.md', '').replace('_', ' ').title()
                    
                    for chunk in file_chunks:
                        try:
                            crystallized = self.contextualizer.crystallize(
                                chunk_text=chunk.text,
                                doc_title=doc_title,
                                header_path=chunk.header_path,
                                generate_summary=True
                            )
                            chunk.crystallized_text = crystallized.crystallized_text
                            chunk.summary = crystallized.summary
                            chunks_crystallized += 1
                        except Exception as e:
                            logger.warning(f"Crystallization failed for chunk: {e}")
                            chunk.crystallized_text = chunk.text
                            chunk.summary = ""
                
                logger.info(f"Crystallized {chunks_crystallized} chunks")
            except Exception as e:
                logger.error(f"Crystallization step failed: {e}")
                errors.append(f"Crystallization error: {e}")
        
        # Step 4: Graph Edge Extraction (Atlas-Hyperion v3.0)
        total_edges = 0
        if self.enable_graph_extraction:
            try:
                logger.info("Extracting graph edges for multi-hop retrieval...")
                for chunk in chunks:
                    edges = self.graph_extractor.extract_edges_simple(chunk.text)
                    chunk.edges = edges
                    total_edges += len(edges)
                
                logger.info(f"Extracted {total_edges} edges from {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Graph extraction failed: {e}")
                errors.append(f"Graph extraction error: {e}")
        
        # Step 5: Generate embeddings
        try:
            # Use crystallized text if available, otherwise use header_path + text
            texts = []
            for chunk in chunks:
                if chunk.crystallized_text:
                    texts.append(chunk.crystallized_text)
                else:
                    texts.append(f"{chunk.header_path}\n{chunk.text}")
            
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embedding_results = self.embedder.embed_texts(texts, show_progress=True)
            embeddings = [r.embedding for r in embedding_results]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            errors.append(f"Embedding error: {e}")
            return IngestionResult(files_processed, len(chunks), 0, errors, chunks_crystallized, total_edges)
        
        # Step 6: Upsert to vector store
        try:
            upserted = self.vector_store.upsert_chunks(chunks, embeddings)
            logger.info(f"Upserted {upserted} chunks to vector store")
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            errors.append(f"Upsert error: {e}")
            return IngestionResult(files_processed, len(chunks), 0, errors, chunks_crystallized, total_edges)
        
        return IngestionResult(
            files_processed=files_processed,
            chunks_created=len(chunks),
            chunks_upserted=upserted,
            errors=errors,
            chunks_crystallized=chunks_crystallized,
            edges_extracted=total_edges
        )
    
    def process_single_file(self, file_path: str) -> IngestionResult:
        """
        Process a single Markdown file.
        Useful for incremental updates.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            IngestionResult
        """
        # Delete existing chunks from this file
        file_name = os.path.basename(file_path)
        try:
            self.vector_store.delete_by_file(file_name)
            logger.info(f"Deleted existing chunks for: {file_name}")
        except Exception as e:
            logger.warning(f"Could not delete existing chunks: {e}")
        
        # Process the file
        return self.run(file_path=file_path)
    
    def get_status(self) -> dict:
        """Get pipeline and vector store status."""
        return {
            "config": {
                "qdrant_url": self.config.qdrant_url,
                "collection": self.config.collection_name,
                "embedding_model": self.config.embedding_model,
                "embedding_dimension": self.config.embedding_dimension
            },
            "vector_store": self.vector_store.get_collection_info(),
            "health": self.vector_store.health_check()
        }


def main():
    """CLI entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Atlas-Hyperion v3.0 Data Ingestion Pipeline"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        help="Directory containing Markdown files"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Single Markdown file to process"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the collection (deletes existing data)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock components (for testing)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pipeline status"
    )
    parser.add_argument(
        "--no-crystallization",
        action="store_true",
        help="Disable contextual crystallization"
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable graph edge extraction"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Initialize pipeline
    pipeline = IngestionPipeline(
        use_mock=args.mock,
        enable_crystallization=not args.no_crystallization,
        enable_graph_extraction=not args.no_graph
    )
    
    if args.status:
        import json
        status = pipeline.get_status()
        print(json.dumps(status, indent=2))
        return
    
    # Run pipeline
    result = pipeline.run(
        source_dir=args.source_dir,
        file_path=args.file,
        recreate_collection=args.recreate
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("ATLAS-HYPERION v3.0 INGESTION COMPLETE")
    print("=" * 50)
    print(f"Files processed:      {result.files_processed}")
    print(f"Chunks created:       {result.chunks_created}")
    print(f"Chunks crystallized:  {result.chunks_crystallized}")
    print(f"Edges extracted:      {result.edges_extracted}")
    print(f"Chunks upserted:      {result.chunks_upserted}")
    print(f"Success:              {result.success}")
    
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()

