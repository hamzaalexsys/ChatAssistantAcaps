"""
Agentic Planner for Atlas-Hyperion v3.0
Self-RAG style decision engine for retrieval strategy.

Decides:
- Whether to retrieve (NO_RETRIEVE for greetings, simple facts)
- How much to retrieve (k parameter)
- Whether to use graph expansion (for complex, multi-hop queries)
- Whether to verify/reflect on the response
"""
import re
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PlannerAction(Enum):
    """Actions the planner can take."""
    NO_RETRIEVE = "no_retrieve"           # Skip retrieval entirely
    RETRIEVE_SIMPLE = "retrieve_simple"   # Standard vector search
    RETRIEVE_GRAPH = "retrieve_graph"     # Graph-enhanced multi-hop
    RETRIEVE_HYBRID = "retrieve_hybrid"   # Hybrid semantic + keyword
    TOOL_CALL = "tool_call"               # Call external tool/API
    REFLECT = "reflect"                   # Needs verification


@dataclass
class PlannerContext:
    """Context for planner decisions."""
    user_id: Optional[str] = None
    session_history: Optional[List[str]] = None
    domain: str = "default"
    locale: str = "fr"
    cache_stats: Optional[Dict[str, Any]] = None


@dataclass
class PlannerDecision:
    """Decision made by the planner."""
    action: PlannerAction
    top_k: int = 5
    graph_depth: int = 2
    confidence_threshold: float = 0.75
    should_verify: bool = True
    reasoning: str = ""
    
    
class AgenticPlanner:
    """
    Self-RAG style planner that decides retrieval strategy.
    
    Features:
    - Query classification (greeting, simple, complex, multi-hop)
    - Adaptive retrieval parameters
    - Confidence-based verification triggers
    """
    
    # Patterns for query classification
    GREETING_PATTERNS = [
        r"^(?:bonjour|salut|hello|hi|hey|coucou|salam|bonsoir)",
        r"^(?:good\s+(?:morning|afternoon|evening))",
        r"^(?:comment\s+(?:ça\s+va|vas-tu|allez-vous))",
        r"^(?:how\s+are\s+you)",
    ]
    
    SIMPLE_QUERY_PATTERNS = [
        r"^(?:qu'?est[- ]ce\s+que|what\s+is|what's)\s+",
        r"^(?:où|where)\s+(?:est|is|are)",
        r"^(?:qui|who)\s+(?:est|is)",
        r"^(?:quand|when)\s+",
    ]
    
    COMPLEX_QUERY_PATTERNS = [
        r"(?:et|and|aussi|also|ainsi\s+que)",
        r"(?:mais|but|cependant|however)",
        r"(?:si|if|dans\s+le\s+cas|in\s+case)",
        r"(?:compare|différence|difference|versus|vs)",
        r"(?:pourquoi|why|comment|how)\s+.{20,}",
    ]
    
    MULTI_HOP_PATTERNS = [
        r"(?:article|section|chapitre)\s+\d+\s+(?:et|and|ou|or)\s+(?:article|section|chapitre)\s+\d+",
        r"(?:selon|according\s+to)\s+.+\s+(?:et|and)\s+",
        r"(?:référence|référé|refer|reference)",
        r"(?:exception|sauf|except|unless)",
        r"(?:conformément|pursuant|tel\s+que\s+(?:défini|prévu))",
    ]
    
    def __init__(
        self,
        default_top_k: int = 5,
        default_graph_depth: int = 2,
        default_confidence_threshold: float = 0.75,
        enable_verification: bool = True
    ):
        """
        Initialize the agentic planner.
        
        Args:
            default_top_k: Default number of documents to retrieve
            default_graph_depth: Default graph traversal depth
            default_confidence_threshold: Threshold for low-confidence flagging
            enable_verification: Whether to enable response verification
        """
        self.default_top_k = default_top_k
        self.default_graph_depth = default_graph_depth
        self.default_confidence_threshold = default_confidence_threshold
        self.enable_verification = enable_verification
        
        # Compile patterns
        self._greeting_re = [re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS]
        self._simple_re = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_QUERY_PATTERNS]
        self._complex_re = [re.compile(p, re.IGNORECASE) for p in self.COMPLEX_QUERY_PATTERNS]
        self._multi_hop_re = [re.compile(p, re.IGNORECASE) for p in self.MULTI_HOP_PATTERNS]
        
        logger.info("AgenticPlanner initialized")
    
    def _is_greeting(self, query: str) -> bool:
        """Check if query is a greeting."""
        clean = query.strip().lower()
        for pattern in self._greeting_re:
            if pattern.search(clean):
                return True
        return len(clean.split()) <= 3 and any(
            greet in clean for greet in ["bonjour", "salut", "hello", "hi", "hey"]
        )
    
    def _is_simple_query(self, query: str) -> bool:
        """Check if query is simple (single-hop)."""
        for pattern in self._simple_re:
            if pattern.search(query):
                return True
        # Short queries without complex patterns
        return len(query.split()) < 10 and not any(
            p.search(query) for p in self._complex_re
        )
    
    def _is_complex_query(self, query: str) -> bool:
        """Check if query is complex (needs more context)."""
        for pattern in self._complex_re:
            if pattern.search(query):
                return True
        return len(query.split()) > 15
    
    def _needs_graph_expansion(self, query: str) -> bool:
        """Check if query needs multi-hop graph expansion."""
        for pattern in self._multi_hop_re:
            if pattern.search(query):
                return True
        # Multiple article/section references
        article_matches = re.findall(r"article\s+\d+", query, re.IGNORECASE)
        return len(article_matches) > 1
    
    def _estimate_query_difficulty(self, query: str) -> float:
        """
        Estimate query difficulty (0.0 = trivial, 1.0 = very complex).
        
        Used to adjust retrieval parameters.
        """
        score = 0.0
        
        # Length factor
        word_count = len(query.split())
        if word_count > 20:
            score += 0.3
        elif word_count > 10:
            score += 0.1
        
        # Complexity indicators
        if self._is_complex_query(query):
            score += 0.3
        
        if self._needs_graph_expansion(query):
            score += 0.2
        
        # Multiple question marks
        if query.count('?') > 1:
            score += 0.1
        
        return min(1.0, score)
    
    def plan(
        self,
        query: str,
        context: Optional[PlannerContext] = None
    ) -> PlannerDecision:
        """
        Plan the retrieval strategy for a query.
        
        Args:
            query: User's query
            context: Optional context for planning
            
        Returns:
            PlannerDecision with action and parameters
        """
        context = context or PlannerContext()
        
        # Check for greeting
        if self._is_greeting(query):
            logger.info(f"Planner: Greeting detected, skipping retrieval")
            return PlannerDecision(
                action=PlannerAction.NO_RETRIEVE,
                top_k=0,
                graph_depth=0,
                should_verify=False,
                reasoning="Query is a greeting, no retrieval needed"
            )
        
        # Estimate difficulty
        difficulty = self._estimate_query_difficulty(query)
        logger.debug(f"Query difficulty estimate: {difficulty:.2f}")
        
        # Determine action
        if self._needs_graph_expansion(query):
            action = PlannerAction.RETRIEVE_GRAPH
            top_k = min(10, self.default_top_k + 3)
            graph_depth = self.default_graph_depth
            reasoning = "Multi-hop query detected, using graph expansion"
        elif self._is_complex_query(query):
            action = PlannerAction.RETRIEVE_HYBRID
            top_k = min(8, self.default_top_k + 2)
            graph_depth = 1
            reasoning = "Complex query, using hybrid retrieval"
        elif self._is_simple_query(query):
            action = PlannerAction.RETRIEVE_SIMPLE
            top_k = self.default_top_k
            graph_depth = 0
            reasoning = "Simple query, using standard retrieval"
        else:
            action = PlannerAction.RETRIEVE_HYBRID
            top_k = self.default_top_k
            graph_depth = 1
            reasoning = "Standard query, using hybrid retrieval"
        
        # Adjust confidence threshold based on difficulty
        confidence_threshold = self.default_confidence_threshold
        if difficulty > 0.5:
            confidence_threshold = min(0.9, confidence_threshold + 0.1)
        
        # Determine if verification is needed
        should_verify = self.enable_verification and difficulty > 0.3
        
        decision = PlannerDecision(
            action=action,
            top_k=top_k,
            graph_depth=graph_depth,
            confidence_threshold=confidence_threshold,
            should_verify=should_verify,
            reasoning=reasoning
        )
        
        logger.info(f"Planner decision: {action.value} (k={top_k}, depth={graph_depth}, verify={should_verify})")
        return decision
    
    def should_reflect(
        self,
        response: str,
        confidence: float,
        decision: PlannerDecision
    ) -> Tuple[bool, str]:
        """
        Determine if response needs reflection/verification.
        
        Args:
            response: Generated response
            confidence: Retrieval confidence score
            decision: Original planner decision
            
        Returns:
            Tuple of (should_reflect, reason)
        """
        if not self.enable_verification:
            return False, "Verification disabled"
        
        # Low confidence
        if confidence < decision.confidence_threshold:
            return True, f"Low retrieval confidence ({confidence:.2f} < {decision.confidence_threshold:.2f})"
        
        # Very short response might indicate issues
        if len(response.split()) < 10:
            return True, "Response is unusually short"
        
        # Hedging language
        hedging_patterns = [
            r"\bi think\b",
            r"\bi believe\b",
            r"\bprobably\b",
            r"\bmaybe\b",
            r"\bmight be\b",
            r"\bcould be\b",
            r"\bpossibly\b",
        ]
        
        for pattern in hedging_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True, f"Hedging language detected: {pattern}"
        
        return False, "Response appears confident"
    
    def replan_on_failure(
        self,
        original_decision: PlannerDecision,
        failure_reason: str
    ) -> PlannerDecision:
        """
        Replan after a failed attempt.
        
        Args:
            original_decision: The decision that led to failure
            failure_reason: Why the attempt failed
            
        Returns:
            New PlannerDecision with adjusted parameters
        """
        logger.info(f"Replanning due to: {failure_reason}")
        
        # Escalate retrieval strategy
        if original_decision.action == PlannerAction.RETRIEVE_SIMPLE:
            new_action = PlannerAction.RETRIEVE_HYBRID
        elif original_decision.action == PlannerAction.RETRIEVE_HYBRID:
            new_action = PlannerAction.RETRIEVE_GRAPH
        else:
            new_action = original_decision.action
        
        # Increase top_k
        new_top_k = min(15, original_decision.top_k + 3)
        
        # Increase graph depth
        new_depth = min(3, original_decision.graph_depth + 1)
        
        return PlannerDecision(
            action=new_action,
            top_k=new_top_k,
            graph_depth=new_depth,
            confidence_threshold=original_decision.confidence_threshold,
            should_verify=True,
            reasoning=f"Replanning after failure: {failure_reason}"
        )


class MockPlanner:
    """Mock planner for testing."""
    
    def plan(self, query: str, context: Optional[PlannerContext] = None) -> PlannerDecision:
        """Always return simple retrieval."""
        return PlannerDecision(
            action=PlannerAction.RETRIEVE_SIMPLE,
            top_k=5,
            graph_depth=0,
            should_verify=False,
            reasoning="Mock planner"
        )
    
    def should_reflect(self, response: str, confidence: float, decision: PlannerDecision) -> Tuple[bool, str]:
        return False, "Mock planner"
    
    def replan_on_failure(self, original_decision: PlannerDecision, failure_reason: str) -> PlannerDecision:
        return original_decision


# Singleton instance
_planner_instance: Optional[AgenticPlanner] = None


def get_planner(use_mock: bool = False) -> AgenticPlanner:
    """Get or create planner instance."""
    global _planner_instance
    if _planner_instance is None:
        if use_mock:
            _planner_instance = MockPlanner()
        else:
            _planner_instance = AgenticPlanner()
    return _planner_instance

