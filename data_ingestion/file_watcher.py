"""
File Watcher Service for Atlas-RAG
Automatically re-indexes documents when they are created, modified, or deleted.
Uses the watchdog library for cross-platform file system monitoring.
"""
import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_ingestion.pipeline import IngestionPipeline
from data_ingestion.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DocumentEventHandler(FileSystemEventHandler):
    """
    Handles file system events for Markdown documents.
    Triggers re-indexing when documents are created, modified, or deleted.
    """
    
    def __init__(
        self,
        pipeline: IngestionPipeline,
        debounce_seconds: float = 2.0,
        file_extensions: tuple = (".md", ".markdown")
    ):
        """
        Initialize the event handler.
        
        Args:
            pipeline: IngestionPipeline instance for processing
            debounce_seconds: Minimum time between processing same file
            file_extensions: Tuple of file extensions to watch
        """
        super().__init__()
        self.pipeline = pipeline
        self.debounce_seconds = debounce_seconds
        self.file_extensions = file_extensions
        self._last_processed: dict = {}  # file_path -> timestamp
        
        logger.info(f"DocumentEventHandler initialized (debounce={debounce_seconds}s)")
    
    def _should_process(self, file_path: str) -> bool:
        """
        Check if file should be processed (debounce logic).
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file should be processed
        """
        # Check file extension
        if not file_path.lower().endswith(self.file_extensions):
            return False
        
        # Check debounce
        now = time.time()
        last_time = self._last_processed.get(file_path, 0)
        
        if now - last_time < self.debounce_seconds:
            logger.debug(f"Debouncing: {file_path}")
            return False
        
        self._last_processed[file_path] = now
        return True
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not self._should_process(file_path):
            return
        
        logger.info(f"📄 New file detected: {os.path.basename(file_path)}")
        self._process_file(file_path)
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not self._should_process(file_path):
            return
        
        logger.info(f"✏️ File modified: {os.path.basename(file_path)}")
        self._process_file(file_path)
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not file_path.lower().endswith(self.file_extensions):
            return
        
        file_name = os.path.basename(file_path)
        logger.info(f"🗑️ File deleted: {file_name}")
        
        try:
            self.pipeline.vector_store.delete_by_file(file_name)
            logger.info(f"✅ Removed vectors for: {file_name}")
        except Exception as e:
            logger.error(f"❌ Failed to remove vectors: {e}")
    
    def on_moved(self, event: FileSystemEvent):
        """Handle file move/rename."""
        if event.is_directory:
            return
        
        src_path = event.src_path
        dest_path = event.dest_path
        
        # Delete old file vectors
        if src_path.lower().endswith(self.file_extensions):
            old_name = os.path.basename(src_path)
            logger.info(f"📦 File moved from: {old_name}")
            try:
                self.pipeline.vector_store.delete_by_file(old_name)
            except Exception as e:
                logger.error(f"Failed to delete old vectors: {e}")
        
        # Index new file
        if dest_path.lower().endswith(self.file_extensions):
            logger.info(f"📦 File moved to: {os.path.basename(dest_path)}")
            self._process_file(dest_path)
    
    def _process_file(self, file_path: str):
        """
        Process a single file through the ingestion pipeline.
        
        Args:
            file_path: Path to the file to process
        """
        try:
            # Use process_single_file which handles delete + re-index
            result = self.pipeline.process_single_file(file_path)
            
            if result.success:
                logger.info(
                    f"✅ Indexed: {os.path.basename(file_path)} "
                    f"({result.chunks_upserted} chunks)"
                )
            else:
                logger.error(f"❌ Failed to index: {result.errors}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")


class FileWatcherService:
    """
    Main file watcher service.
    Monitors a directory for changes and triggers document re-indexing.
    """
    
    def __init__(
        self,
        watch_dir: Optional[str] = None,
        use_mock: bool = False,
        debounce_seconds: float = 2.0
    ):
        """
        Initialize the file watcher service.
        
        Args:
            watch_dir: Directory to watch (default: config documents_dir)
            use_mock: Use mock components for testing
            debounce_seconds: Debounce time for file events
        """
        self.config = get_config()
        self.watch_dir = watch_dir or self.config.documents_dir
        self.use_mock = use_mock
        self.debounce_seconds = debounce_seconds
        
        # Initialize pipeline
        self.pipeline = IngestionPipeline(use_mock=use_mock)
        
        # Initialize event handler and observer
        self.event_handler = DocumentEventHandler(
            pipeline=self.pipeline,
            debounce_seconds=debounce_seconds
        )
        self.observer = Observer()
        
        logger.info(f"FileWatcherService initialized")
        logger.info(f"  Watch directory: {self.watch_dir}")
        logger.info(f"  Mock mode: {use_mock}")
    
    def start(self):
        """Start watching the directory."""
        if not os.path.isdir(self.watch_dir):
            raise NotADirectoryError(f"Watch directory not found: {self.watch_dir}")
        
        self.observer.schedule(
            self.event_handler,
            self.watch_dir,
            recursive=False  # Don't watch subdirectories
        )
        
        self.observer.start()
        logger.info(f"👁️ Started watching: {self.watch_dir}")
        logger.info("Press Ctrl+C to stop...")
    
    def stop(self):
        """Stop watching."""
        self.observer.stop()
        self.observer.join()
        logger.info("🛑 File watcher stopped")
    
    def run_forever(self):
        """Run the watcher until interrupted."""
        self.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n⏹️ Stopping file watcher...")
            self.stop()
    
    def get_status(self) -> dict:
        """Get watcher status."""
        return {
            "watch_dir": self.watch_dir,
            "is_alive": self.observer.is_alive() if hasattr(self, 'observer') else False,
            "pipeline_status": self.pipeline.get_status(),
            "debounce_seconds": self.debounce_seconds
        }


def main():
    """CLI entry point for the file watcher service."""
    parser = argparse.ArgumentParser(
        description="Atlas-RAG File Watcher Service - Auto-index documents on change"
    )
    parser.add_argument(
        "--watch-dir",
        type=str,
        help="Directory to watch for changes"
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        help="Debounce time in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock components (for testing)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("""
╔═══════════════════════════════════════════════════════════╗
║           Atlas-RAG File Watcher Service                  ║
║      Auto-index documents when they change                ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Create and run watcher
    watcher = FileWatcherService(
        watch_dir=args.watch_dir,
        use_mock=args.mock,
        debounce_seconds=args.debounce
    )
    
    watcher.run_forever()


if __name__ == "__main__":
    main()

