"""
Guardrails Module for Atlas-Hyperion v3.0
Implements 3-tier input/output validation and hallucination prevention.

Tier 1: Grammar + Pattern-based constraints
Tier 2: Token-level suspicion scoring (entropy-based heuristics)
Tier 3: NLI-based verification (neural entailment)
"""
import re
import logging
import math
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .nli_verifier import NLIVerifier, VerificationResult, get_nli_verifier

logger = logging.getLogger(__name__)


class RailType(Enum):
    """Types of guardrails."""
    INPUT = "input"
    OUTPUT = "output"
    FACT_CHECK = "fact_check"


class BlockReason(Enum):
    """Reasons for blocking a message."""
    JAILBREAK = "jailbreak_attempt"
    TOXICITY = "toxic_content"
    OFF_TOPIC = "off_topic"
    LOW_CONFIDENCE = "low_retrieval_confidence"
    HALLUCINATION = "potential_hallucination"
    PII_DETECTED = "pii_detected"
    # Atlas-Hyperion v3.0: NLI-based reasons
    NLI_CONTRADICTION = "nli_contradiction"
    NLI_UNVERIFIED = "nli_unverified"
    HIGH_SUSPICION = "high_suspicion_score"


@dataclass
class GuardrailResult:
    """Result of guardrail check."""
    passed: bool
    blocked_reason: Optional[BlockReason] = None
    message: Optional[str] = None
    confidence: float = 1.0
    # Atlas-Hyperion v3.0: Additional verification info
    verification_details: Optional[Dict[str, Any]] = None
    tier_results: Optional[Dict[str, bool]] = None  # tier1, tier2, tier3 pass/fail
    correction_prompt: Optional[str] = None  # For self-correction
    
    @property
    def should_block(self) -> bool:
        return not self.passed


class InputGuardrails:
    """
    Input validation guardrails.
    Checks for jailbreaks, toxicity, and off-topic queries.
    """
    
    # Jailbreak patterns
    JAILBREAK_PATTERNS = [
        r"ignore.*(?:previous|your).*instructions",
        r"forget.*(?:previous|your).*instructions",
        r"pretend.*(?:you are|to be)",
        r"act as if",
        r"you are now",
        r"new persona",
        r"bypass.*(?:filters|restrictions)",
        r"DAN.*mode",
        r"developer.*mode",
    ]
    
    # Off-topic patterns
    OFF_TOPIC_PATTERNS = [
        r"(?:what's|what is).*weather",
        r"(?:tell|write).*(?:joke|story|poem)",
        r"(?:who won|score of).*(?:game|match)",
        r"(?:latest|recent).*news",
        r"(?:stock|crypto).*price",
        r"(?:recipe|cook).*",
        r"(?:translate|translation)",
    ]
    
    # Toxic patterns (simplified)
    TOXIC_PATTERNS = [
        r"(?:you are|you're).*(?:stupid|idiot|dumb)",
        r"(?:this is|it's).*(?:garbage|trash|useless)",
        r"\b(?:hate|kill|die)\b",
    ]
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self.jailbreak_re = [re.compile(p, re.IGNORECASE) for p in self.JAILBREAK_PATTERNS]
        self.off_topic_re = [re.compile(p, re.IGNORECASE) for p in self.OFF_TOPIC_PATTERNS]
        self.toxic_re = [re.compile(p, re.IGNORECASE) for p in self.TOXIC_PATTERNS]
    
    def check(self, user_input: str) -> GuardrailResult:
        """
        Check user input against all input guardrails.
        
        Args:
            user_input: The user's message
            
        Returns:
            GuardrailResult indicating pass/fail
        """
        if not self.enabled:
            return GuardrailResult(passed=True)
        
        # Check jailbreak
        result = self._check_jailbreak(user_input)
        if result.should_block:
            logger.warning(f"Jailbreak attempt detected: {user_input[:100]}")
            return result
        
        # Check toxicity
        result = self._check_toxicity(user_input)
        if result.should_block:
            logger.warning(f"Toxic content detected: {user_input[:100]}")
            return result
        
        # Check off-topic
        result = self._check_off_topic(user_input)
        if result.should_block:
            logger.info(f"Off-topic query detected: {user_input[:100]}")
            return result
        
        return GuardrailResult(passed=True)
    
    def _check_jailbreak(self, text: str) -> GuardrailResult:
        """Check for jailbreak attempts."""
        for pattern in self.jailbreak_re:
            if pattern.search(text):
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.JAILBREAK,
                    message="I can only help with questions about ACAPS documentation and regulations."
                )
        return GuardrailResult(passed=True)
    
    def _check_toxicity(self, text: str) -> GuardrailResult:
        """Check for toxic content."""
        for pattern in self.toxic_re:
            if pattern.search(text):
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.TOXICITY,
                    message="Please rephrase your question in a respectful manner."
                )
        return GuardrailResult(passed=True)
    
    def _check_off_topic(self, text: str) -> GuardrailResult:
        """Check for off-topic queries."""
        for pattern in self.off_topic_re:
            if pattern.search(text):
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.OFF_TOPIC,
                    message="I'm designed to assist with ACAPS regulations, internal rules, and site usage. I cannot help with topics outside this scope."
                )
        return GuardrailResult(passed=True)


class OutputGuardrails:
    """
    Output validation guardrails with 3-tier verification (Atlas-Hyperion v3.0).
    
    Tier 1: Pattern-based hallucination detection
    Tier 2: Token-level suspicion scoring
    Tier 3: NLI-based verification (optional)
    """
    
    # Tier 1: Phrases indicating potential hallucination
    HALLUCINATION_INDICATORS = [
        r"I think",
        r"I believe",
        r"probably",
        r"might be",
        r"could be",
        r"as far as I know",
        r"in my opinion",
        r"generally speaking",
    ]
    
    # Tier 2: High-uncertainty patterns (for suspicion scoring)
    UNCERTAINTY_PATTERNS = [
        (r"\bseems?\b", 0.2),
        (r"\bapparently\b", 0.3),
        (r"\bpossibly\b", 0.3),
        (r"\blikely\b", 0.2),
        (r"\bunlikely\b", 0.2),
        (r"\bmaybe\b", 0.3),
        (r"\bperhaps\b", 0.3),
        (r"\bsomehow\b", 0.2),
        (r"\bestimate\b", 0.2),
        (r"\bguess\b", 0.4),
    ]
    
    def __init__(
        self,
        enabled: bool = True,
        confidence_threshold: float = 0.75,
        suspicion_threshold: float = 0.5,
        enable_nli: bool = True,
        use_mock_nli: bool = False
    ):
        """
        Initialize output guardrails.
        
        Args:
            enabled: Enable guardrails
            confidence_threshold: Minimum retrieval confidence
            suspicion_threshold: Tier 2 suspicion threshold
            enable_nli: Enable Tier 3 NLI verification
            use_mock_nli: Use mock NLI verifier
        """
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.suspicion_threshold = suspicion_threshold
        self.enable_nli = enable_nli
        
        self.hallucination_re = [
            re.compile(p, re.IGNORECASE) for p in self.HALLUCINATION_INDICATORS
        ]
        self.uncertainty_patterns = [
            (re.compile(p, re.IGNORECASE), score) 
            for p, score in self.UNCERTAINTY_PATTERNS
        ]
        
        # Initialize NLI verifier for Tier 3
        self.nli_verifier = get_nli_verifier(
            enabled=enable_nli,
            use_mock=use_mock_nli
        ) if enable_nli else None
    
    def check(
        self,
        response: str,
        context: str,
        retrieval_score: float,
        run_nli: bool = True
    ) -> GuardrailResult:
        """
        Check model output with 3-tier verification.
        
        Args:
            response: The model's response
            context: The retrieved context used
            retrieval_score: Similarity score from retrieval
            run_nli: Whether to run Tier 3 NLI verification
            
        Returns:
            GuardrailResult with tier results
        """
        if not self.enabled:
            return GuardrailResult(passed=True)
        
        tier_results = {}
        
        # Check retrieval confidence first
        if retrieval_score < self.confidence_threshold:
            logger.info(f"Low retrieval score: {retrieval_score}")
            return GuardrailResult(
                passed=False,
                blocked_reason=BlockReason.LOW_CONFIDENCE,
                message="I cannot find relevant information in the available documents.",
                confidence=retrieval_score,
                tier_results={"confidence_check": False}
            )
        
        # === TIER 1: Pattern-based checks ===
        tier1_result = self._check_tier1(response, context)
        tier_results["tier1"] = tier1_result.passed
        
        if tier1_result.should_block:
            tier1_result.tier_results = tier_results
            return tier1_result
        
        # === TIER 2: Suspicion scoring ===
        tier2_result = self._check_tier2(response)
        tier_results["tier2"] = tier2_result.passed
        
        if tier2_result.should_block:
            tier2_result.tier_results = tier_results
            return tier2_result
        
        # === TIER 3: NLI verification ===
        if run_nli and self.enable_nli and self.nli_verifier:
            tier3_result = self._check_tier3(response, context)
            tier_results["tier3"] = tier3_result.passed
            
            if tier3_result.should_block:
                tier3_result.tier_results = tier_results
                return tier3_result
        else:
            tier_results["tier3"] = True  # Skipped
        
        return GuardrailResult(
            passed=True, 
            confidence=retrieval_score,
            tier_results=tier_results
        )
    
    def _check_tier1(self, response: str, context: str) -> GuardrailResult:
        """
        Tier 1: Pattern-based hallucination detection.
        
        - Check for hedging phrases
        - Check for ungrounded entities
        """
        # Check for hallucination indicators
        result = self._check_hallucination_phrases(response)
        if result.should_block:
            return result
        
        # Check if response is grounded
        result = self._check_grounding(response, context)
        if result.should_block:
            return result
        
        return GuardrailResult(passed=True)
    
    def _check_tier2(self, response: str) -> GuardrailResult:
        """
        Tier 2: Token-level suspicion scoring.
        
        Calculates aggregate suspicion based on uncertainty patterns.
        """
        total_suspicion = 0.0
        matched_patterns = []
        
        for pattern, score in self.uncertainty_patterns:
            matches = pattern.findall(response)
            if matches:
                total_suspicion += score * len(matches)
                matched_patterns.append((pattern.pattern, len(matches)))
        
        # Normalize by response length (per 100 words)
        word_count = len(response.split())
        if word_count > 0:
            normalized_suspicion = total_suspicion / (word_count / 100)
        else:
            normalized_suspicion = 0.0
        
        if normalized_suspicion > self.suspicion_threshold:
            logger.warning(f"High suspicion score: {normalized_suspicion:.2f} (patterns: {matched_patterns})")
            return GuardrailResult(
                passed=False,
                blocked_reason=BlockReason.HIGH_SUSPICION,
                message="The response contains too many uncertain statements.",
                verification_details={
                    "suspicion_score": normalized_suspicion,
                    "matched_patterns": matched_patterns
                }
            )
        
        return GuardrailResult(
            passed=True,
            verification_details={"suspicion_score": normalized_suspicion}
        )
    
    def _check_tier3(self, response: str, context: str) -> GuardrailResult:
        """
        Tier 3: NLI-based verification.
        
        Uses neural entailment model to verify claims.
        """
        if not self.nli_verifier:
            return GuardrailResult(passed=True)
        
        try:
            verification = self.nli_verifier.verify_response(response, context)
            
            if not verification.passed:
                # Get correction prompt for self-healing
                correction_prompt = self.nli_verifier.get_correction_prompt(
                    response, verification
                )
                
                if verification.contradicted_claims:
                    return GuardrailResult(
                        passed=False,
                        blocked_reason=BlockReason.NLI_CONTRADICTION,
                        message="Some statements contradict the source documents.",
                        verification_details={
                            "overall_score": verification.overall_score,
                            "contradicted_count": len(verification.contradicted_claims),
                            "unverified_count": len(verification.unverified_claims)
                        },
                        correction_prompt=correction_prompt
                    )
                else:
                    return GuardrailResult(
                        passed=False,
                        blocked_reason=BlockReason.NLI_UNVERIFIED,
                        message="Some statements could not be verified against the source documents.",
                        verification_details={
                            "overall_score": verification.overall_score,
                            "unverified_count": len(verification.unverified_claims)
                        },
                        correction_prompt=correction_prompt
                    )
            
            return GuardrailResult(
                passed=True,
                verification_details={
                    "overall_score": verification.overall_score,
                    "verified_claims": len(verification.claims)
                }
            )
            
        except Exception as e:
            logger.error(f"Tier 3 NLI verification failed: {e}")
            # Don't block on NLI failure, just log
            return GuardrailResult(
                passed=True,
                verification_details={"nli_error": str(e)}
            )
    
    def _check_hallucination_phrases(self, response: str) -> GuardrailResult:
        """Check for phrases that indicate uncertainty/hallucination."""
        for pattern in self.hallucination_re:
            if pattern.search(response):
                return GuardrailResult(
                    passed=False,
                    blocked_reason=BlockReason.HALLUCINATION,
                    message="I cannot provide a definitive answer based on the available documents."
                )
        return GuardrailResult(passed=True)
    
    def _check_grounding(self, response: str, context: str) -> GuardrailResult:
        """
        Check if response is grounded in context.
        Simple heuristic: major named entities in response should appear in context.
        """
        # Extract potential entity-like terms (capitalized words)
        response_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response))
        context_lower = context.lower()
        
        # Check if entities are grounded
        ungrounded = []
        for entity in response_entities:
            # Skip common words
            if entity.lower() in ['the', 'this', 'that', 'article', 'section', 'selon', 'according']:
                continue
            if entity.lower() not in context_lower:
                ungrounded.append(entity)
        
        # If more than 2 ungrounded entities, flag as potential hallucination
        if len(ungrounded) > 2:
            logger.warning(f"Ungrounded entities found: {ungrounded}")
            return GuardrailResult(
                passed=False,
                blocked_reason=BlockReason.HALLUCINATION,
                message="I cannot verify this information in the available documents.",
                verification_details={"ungrounded_entities": ungrounded}
            )
        
        return GuardrailResult(passed=True)


class Guardrails:
    """
    Main guardrails orchestrator for Atlas-Hyperion v3.0.
    Combines input validation and 3-tier output verification.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        confidence_threshold: float = 0.75,
        enable_nli: bool = True,
        use_mock: bool = False
    ):
        """
        Initialize guardrails.
        
        Args:
            enabled: Enable all guardrails
            confidence_threshold: Retrieval confidence threshold
            enable_nli: Enable Tier 3 NLI verification
            use_mock: Use mock components for testing
        """
        self.enabled = enabled
        self.enable_nli = enable_nli
        self.input_rails = InputGuardrails(enabled=enabled)
        self.output_rails = OutputGuardrails(
            enabled=enabled,
            confidence_threshold=confidence_threshold,
            enable_nli=enable_nli,
            use_mock_nli=use_mock
        )
        logger.info(f"Guardrails initialized (enabled={enabled}, threshold={confidence_threshold}, nli={enable_nli})")
    
    def check_input(self, user_input: str) -> GuardrailResult:
        """Check user input."""
        return self.input_rails.check(user_input)
    
    def check_output(
        self,
        response: str,
        context: str,
        retrieval_score: float,
        run_nli: bool = True
    ) -> GuardrailResult:
        """
        Check model output with 3-tier verification.
        
        Args:
            response: Model response
            context: Retrieved context
            retrieval_score: Retrieval confidence score
            run_nli: Whether to run Tier 3 NLI verification
        """
        return self.output_rails.check(response, context, retrieval_score, run_nli)
    
    def get_blocked_response(self, reason: BlockReason) -> str:
        """Get appropriate response for blocked content."""
        responses = {
            BlockReason.JAILBREAK: "I can only help with questions about ACAPS documentation and regulations.",
            BlockReason.TOXICITY: "Please rephrase your question in a respectful manner.",
            BlockReason.OFF_TOPIC: "I'm designed to assist with ACAPS regulations, internal rules, and site usage only.",
            BlockReason.LOW_CONFIDENCE: "I cannot find this information in the available documents.",
            BlockReason.HALLUCINATION: "I cannot provide a verified answer based on the available documents.",
            BlockReason.PII_DETECTED: "I cannot process requests containing personal information.",
            # Atlas-Hyperion v3.0 responses
            BlockReason.NLI_CONTRADICTION: "Some information in the response contradicts the source documents. Please try rephrasing your question.",
            BlockReason.NLI_UNVERIFIED: "I cannot fully verify this information against the available documents.",
            BlockReason.HIGH_SUSPICION: "The response contains too many uncertain statements. Please ask a more specific question.",
        }
        return responses.get(reason, "I cannot process this request.")


# Singleton instance
_guardrails_instance: Optional[Guardrails] = None


def get_guardrails(
    enabled: bool = True,
    confidence_threshold: float = 0.75,
    enable_nli: bool = True,
    use_mock: bool = False
) -> Guardrails:
    """Get or create guardrails instance."""
    global _guardrails_instance
    if _guardrails_instance is None:
        _guardrails_instance = Guardrails(
            enabled=enabled,
            confidence_threshold=confidence_threshold,
            enable_nli=enable_nli,
            use_mock=use_mock
        )
    return _guardrails_instance

