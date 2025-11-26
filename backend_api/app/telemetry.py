"""
Telemetry and Evaluation Plane for Atlas-Hyperion v3.0
Logs interactions, tracks metrics, and enables continuous learning.
"""
import logging
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field
from collections import deque
from threading import Lock
import json

logger = logging.getLogger(__name__)


@dataclass
class QueryLog:
    """Log entry for a query interaction."""
    timestamp: str
    query: str
    query_hash: str
    retrieval_ids: List[str]
    retrieval_scores: List[float]
    answer: str
    answer_length: int
    confidence: float
    latency_ms: float
    cache_hit: bool
    cache_level: str
    planner_action: str
    verification_passed: bool
    verification_details: Optional[Dict[str, Any]] = None
    user_feedback: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricsSnapshot:
    """Point-in-time metrics snapshot."""
    timestamp: str
    total_queries: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    verification_pass_rate: float
    planner_actions: Dict[str, int]
    error_count: int


class Telemetry:
    """
    Telemetry system for Atlas-Hyperion v3.0.
    
    Provides:
    - Query logging
    - Metrics aggregation
    - Cache statistics
    - Latency tracking
    - Verification pass rates
    """
    
    def __init__(
        self,
        max_log_entries: int = 10000,
        enabled: bool = True
    ):
        """
        Initialize telemetry.
        
        Args:
            max_log_entries: Maximum log entries to keep in memory
            enabled: Whether telemetry is enabled
        """
        self.enabled = enabled
        self.max_log_entries = max_log_entries
        
        # In-memory logs (circular buffer)
        self._logs: deque = deque(maxlen=max_log_entries)
        self._lock = Lock()
        
        # Counters
        self._total_queries = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._verification_passes = 0
        self._verification_fails = 0
        self._errors = 0
        
        # Latency tracking
        self._latencies: deque = deque(maxlen=1000)
        
        # Planner action distribution
        self._planner_actions: Dict[str, int] = {}
        
        # Start time
        self._start_time = datetime.now(timezone.utc)
        
        logger.info(f"Telemetry initialized (enabled={enabled})")
    
    def _hash_query(self, query: str) -> str:
        """Generate a hash for the query."""
        import hashlib
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
    
    def log_query(
        self,
        query: str,
        retrieval_ids: List[str],
        retrieval_scores: List[float],
        answer: str,
        confidence: float,
        latency_ms: float,
        cache_hit: bool,
        cache_level: str = "",
        planner_action: str = "",
        verification_passed: bool = True,
        verification_details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> QueryLog:
        """
        Log a query interaction.
        
        Args:
            query: User's query
            retrieval_ids: IDs of retrieved documents
            retrieval_scores: Scores of retrieved documents
            answer: Generated answer
            confidence: Confidence score
            latency_ms: Total latency in milliseconds
            cache_hit: Whether cache was hit
            cache_level: Which cache level was hit
            planner_action: Planner decision
            verification_passed: Whether verification passed
            verification_details: Details from verification
            session_id: Optional session identifier
            
        Returns:
            QueryLog entry
        """
        if not self.enabled:
            return None
        
        log_entry = QueryLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query[:500],  # Truncate long queries
            query_hash=self._hash_query(query),
            retrieval_ids=retrieval_ids[:10],  # Limit IDs stored
            retrieval_scores=retrieval_scores[:10],
            answer=answer[:1000],  # Truncate long answers
            answer_length=len(answer),
            confidence=confidence,
            latency_ms=latency_ms,
            cache_hit=cache_hit,
            cache_level=cache_level,
            planner_action=planner_action,
            verification_passed=verification_passed,
            verification_details=verification_details,
            session_id=session_id
        )
        
        with self._lock:
            self._logs.append(log_entry)
            self._total_queries += 1
            
            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            
            if verification_passed:
                self._verification_passes += 1
            else:
                self._verification_fails += 1
            
            self._latencies.append(latency_ms)
            
            if planner_action:
                self._planner_actions[planner_action] = \
                    self._planner_actions.get(planner_action, 0) + 1
        
        return log_entry
    
    def log_error(self, error: str, context: Optional[Dict[str, Any]] = None):
        """Log an error occurrence."""
        if not self.enabled:
            return
        
        with self._lock:
            self._errors += 1
        
        logger.error(f"Telemetry error: {error}", extra={"context": context})
    
    def get_metrics(self) -> MetricsSnapshot:
        """Get current metrics snapshot."""
        with self._lock:
            total = self._total_queries
            cache_hits = self._cache_hits
            cache_misses = self._cache_misses
            verification_passes = self._verification_passes
            verification_fails = self._verification_fails
            latencies = list(self._latencies)
            planner_actions = dict(self._planner_actions)
            errors = self._errors
        
        # Calculate rates
        cache_hit_rate = cache_hits / total if total > 0 else 0.0
        verification_total = verification_passes + verification_fails
        verification_pass_rate = verification_passes / verification_total if verification_total > 0 else 1.0
        
        # Calculate latency percentiles
        if latencies:
            sorted_latencies = sorted(latencies)
            avg_latency = sum(latencies) / len(latencies)
            p50_idx = int(len(sorted_latencies) * 0.5)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            
            p50_latency = sorted_latencies[p50_idx] if p50_idx < len(sorted_latencies) else 0
            p95_latency = sorted_latencies[p95_idx] if p95_idx < len(sorted_latencies) else 0
            p99_latency = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0.0
        
        return MetricsSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_queries=total,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hit_rate=round(cache_hit_rate, 4),
            avg_latency_ms=round(avg_latency, 2),
            p50_latency_ms=round(p50_latency, 2),
            p95_latency_ms=round(p95_latency, 2),
            p99_latency_ms=round(p99_latency, 2),
            verification_pass_rate=round(verification_pass_rate, 4),
            planner_actions=planner_actions,
            error_count=errors
        )
    
    def get_prometheus_metrics(self) -> str:
        """
        Get metrics in Prometheus format.
        
        Returns:
            Prometheus-style metrics string
        """
        metrics = self.get_metrics()
        
        lines = [
            "# HELP atlas_queries_total Total number of queries processed",
            "# TYPE atlas_queries_total counter",
            f"atlas_queries_total {metrics.total_queries}",
            "",
            "# HELP atlas_cache_hits_total Total cache hits",
            "# TYPE atlas_cache_hits_total counter",
            f"atlas_cache_hits_total {metrics.cache_hits}",
            "",
            "# HELP atlas_cache_hit_rate Cache hit rate",
            "# TYPE atlas_cache_hit_rate gauge",
            f"atlas_cache_hit_rate {metrics.cache_hit_rate}",
            "",
            "# HELP atlas_latency_ms Query latency in milliseconds",
            "# TYPE atlas_latency_ms summary",
            f'atlas_latency_ms{{quantile="0.5"}} {metrics.p50_latency_ms}',
            f'atlas_latency_ms{{quantile="0.95"}} {metrics.p95_latency_ms}',
            f'atlas_latency_ms{{quantile="0.99"}} {metrics.p99_latency_ms}',
            f"atlas_latency_ms_avg {metrics.avg_latency_ms}",
            "",
            "# HELP atlas_verification_pass_rate Verification pass rate",
            "# TYPE atlas_verification_pass_rate gauge",
            f"atlas_verification_pass_rate {metrics.verification_pass_rate}",
            "",
            "# HELP atlas_errors_total Total error count",
            "# TYPE atlas_errors_total counter",
            f"atlas_errors_total {metrics.error_count}",
            "",
        ]
        
        # Planner action distribution
        lines.append("# HELP atlas_planner_actions Planner action distribution")
        lines.append("# TYPE atlas_planner_actions counter")
        for action, count in metrics.planner_actions.items():
            lines.append(f'atlas_planner_actions{{action="{action}"}} {count}')
        
        return "\n".join(lines)
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query logs."""
        with self._lock:
            logs = list(self._logs)[-limit:]
        
        return [log.to_dict() for log in logs]
    
    def add_user_feedback(
        self,
        query_hash: str,
        feedback: str
    ) -> bool:
        """
        Add user feedback to a logged query.
        
        Args:
            query_hash: Hash of the query to update
            feedback: Feedback string (thumbs_up, thumbs_down, correction)
            
        Returns:
            True if feedback was added
        """
        with self._lock:
            for log in reversed(self._logs):
                if log.query_hash == query_hash:
                    log.user_feedback = feedback
                    return True
        return False
    
    def reset(self):
        """Reset all telemetry data."""
        with self._lock:
            self._logs.clear()
            self._total_queries = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._verification_passes = 0
            self._verification_fails = 0
            self._errors = 0
            self._latencies.clear()
            self._planner_actions.clear()
        
        logger.info("Telemetry reset")
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()


# Singleton instance
_telemetry_instance: Optional[Telemetry] = None


def get_telemetry(enabled: bool = True) -> Telemetry:
    """Get or create telemetry instance."""
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = Telemetry(enabled=enabled)
    return _telemetry_instance

