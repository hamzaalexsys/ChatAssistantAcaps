"""
Guardrails Module for Atlas-RAG
Implements input/output validation and hallucination prevention.
"""
import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

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


@dataclass
class GuardrailResult:
    """Result of guardrail check."""
    passed: bool
    blocked_reason: Optional[BlockReason] = None
    message: Optional[str] = None
    confidence: float = 1.0
    
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
    Output validation guardrails.
    Checks for hallucinations and ensures factual consistency.
    """
    
    # Phrases indicating potential hallucination
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
    
    def __init__(
        self,
        enabled: bool = True,
        confidence_threshold: float = 0.75
    ):
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.hallucination_re = [
            re.compile(p, re.IGNORECASE) for p in self.HALLUCINATION_INDICATORS
        ]
    
    def check(
        self,
        response: str,
        context: str,
        retrieval_score: float
    ) -> GuardrailResult:
        """
        Check model output for hallucinations.
        
        Args:
            response: The model's response
            context: The retrieved context used
            retrieval_score: Similarity score from retrieval
            
        Returns:
            GuardrailResult indicating pass/fail
        """
        if not self.enabled:
            return GuardrailResult(passed=True)
        
        # Check retrieval confidence
        if retrieval_score < self.confidence_threshold:
            logger.info(f"Low retrieval score: {retrieval_score}")
            return GuardrailResult(
                passed=False,
                blocked_reason=BlockReason.LOW_CONFIDENCE,
                message="I cannot find relevant information in the available documents.",
                confidence=retrieval_score
            )
        
        # Check for hallucination indicators
        result = self._check_hallucination_phrases(response)
        if result.should_block:
            return result
        
        # Check if response references content not in context
        result = self._check_grounding(response, context)
        if result.should_block:
            return result
        
        return GuardrailResult(passed=True, confidence=retrieval_score)
    
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
        hhhh heuristic simple hada, khass n9ado
        """
        # TODO : N9AD HAD TKHARBI9 and I Implement a more robust grounding check.
        # Extract potential entity-like terms (capitalized words)
        # print("response:", response)
        # print("Context:", context)

        # response_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response))
        # context_lower = context.lower()
        # print("Response entities:", response_entities)
        # # Check if entities are grounded
        # ungrounded = []
        # for entity in response_entities:
        #     # Skip common words
        #     if entity.lower() in ['the', 'this', 'that', 'article', 'section']:
        #         continue
        #     if entity.lower() not in context_lower:
        #         ungrounded.append(entity)
        
        # # If more than 2 ungrounded entities, flag as potential hallucination
        # if len(ungrounded) > 2:
        #     logger.warning(f"Ungrounded entities found: {ungrounded}")
        #     return GuardrailResult(
        #         passed=False,
        #         blocked_reason=BlockReason.HALLUCINATION,
        #         message="I cannot verify this information in the available documents."
        #     )
        
        return GuardrailResult(passed=True)


class Guardrails:
    """
    Main guardrails orchestrator.
    Combines input and output validation.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        confidence_threshold: float = 0.75
    ):
        self.enabled = enabled
        self.input_rails = InputGuardrails(enabled=enabled)
        self.output_rails = OutputGuardrails(
            enabled=enabled,
            confidence_threshold=confidence_threshold
        )
        logger.info(f"Guardrails initialized (enabled={enabled}, threshold={confidence_threshold})")
    
    def check_input(self, user_input: str) -> GuardrailResult:
        """Check user input."""
        return self.input_rails.check(user_input)
    
    def check_output(
        self,
        response: str,
        context: str,
        retrieval_score: float
    ) -> GuardrailResult:
        """Check model output."""
        return self.output_rails.check(response, context, retrieval_score)
    
    def get_blocked_response(self, reason: BlockReason) -> str:
        """Get appropriate response for blocked content."""
        responses = {
            BlockReason.JAILBREAK: "I can only help with questions about ACAPS documentation and regulations.",
            BlockReason.TOXICITY: "Please rephrase your question in a respectful manner.",
            BlockReason.OFF_TOPIC: "I'm designed to assist with ACAPS regulations, internal rules, and site usage only.",
            BlockReason.LOW_CONFIDENCE: "I cannot find this information in the available documents.",
            BlockReason.HALLUCINATION: "I cannot provide a verified answer based on the available documents.",
            BlockReason.PII_DETECTED: "I cannot process requests containing personal information.",
        }
        return responses.get(reason, "I cannot process this request.")


# Singleton instance
_guardrails_instance: Optional[Guardrails] = None


def get_guardrails(
    enabled: bool = True,
    confidence_threshold: float = 0.75
) -> Guardrails:
    """Get or create guardrails instance."""
    global _guardrails_instance
    if _guardrails_instance is None:
        _guardrails_instance = Guardrails(
            enabled=enabled,
            confidence_threshold=confidence_threshold
        )
    return _guardrails_instance

