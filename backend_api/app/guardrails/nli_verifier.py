"""
NLI-Based Hallucination Verifier for Atlas-Hyperion v3.0
Uses Neural Language Inference to verify claims against context.

Three-tier verification:
- Tier 1: Grammar + JSON constraints (existing guardrails)
- Tier 2: Token-level suspicion score (entropy-based)
- Tier 3: NLI verification (this module)
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EntailmentLabel(Enum):
    """NLI entailment labels."""
    ENTAILS = "entailment"
    CONTRADICTS = "contradiction"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass
class Claim:
    """An atomic claim extracted from a response."""
    text: str
    start_idx: int
    end_idx: int
    source_sentence: str
    

@dataclass 
class ClaimVerification:
    """Verification result for a single claim."""
    claim: Claim
    label: EntailmentLabel
    score: float
    supporting_context: str
    explanation: str


@dataclass
class VerificationResult:
    """Overall verification result."""
    passed: bool
    claims: List[ClaimVerification]
    overall_score: float
    unverified_claims: List[Claim]
    contradicted_claims: List[Claim]
    explanation: str
    

class NLIVerifier:
    """
    Neural Language Inference verifier for hallucination detection.
    
    Uses a cross-encoder NLI model to check if response claims
    are entailed by the retrieved context.
    """
    
    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        entailment_threshold: float = 0.7,
        contradiction_threshold: float = 0.7,
        enabled: bool = True,
        use_mock: bool = False
    ):
        """
        Initialize NLI verifier.
        
        Args:
            model_name: HuggingFace model name for NLI
            entailment_threshold: Minimum score for entailment
            contradiction_threshold: Minimum score for contradiction
            enabled: Whether verification is enabled
            use_mock: Use mock verification (no model loading)
        """
        self.model_name = model_name
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.enabled = enabled
        self.use_mock = use_mock
        self._model = None
        
        logger.info(f"NLIVerifier initialized (enabled={enabled}, mock={use_mock})")
    
    @property
    def model(self):
        """Lazy load NLI model."""
        if self._model is None and not self.use_mock:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading NLI model: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
                logger.info("NLI model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load NLI model: {e}")
                self._model = None
        return self._model
    
    def _extract_claims(self, response: str) -> List[Claim]:
        """
        Extract atomic claims from response text.
        
        Simple sentence-based extraction with filtering.
        """
        claims = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', response)
        
        current_idx = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Skip very short sentences (likely not claims)
            if len(sentence.split()) < 4:
                current_idx += len(sentence) + 1
                continue
            
            # Skip questions
            if sentence.endswith('?'):
                current_idx += len(sentence) + 1
                continue
            
            # Skip meta-sentences
            meta_patterns = [
                r"^(?:je ne peux pas|i cannot|i can't)",
                r"^(?:je suis|i am)",
                r"^(?:based on|selon|d'après)",
                r"^(?:please|s'il vous plaît|veuillez)",
            ]
            
            is_meta = any(re.match(p, sentence, re.IGNORECASE) for p in meta_patterns)
            if is_meta:
                current_idx += len(sentence) + 1
                continue
            
            claims.append(Claim(
                text=sentence,
                start_idx=current_idx,
                end_idx=current_idx + len(sentence),
                source_sentence=sentence
            ))
            
            current_idx += len(sentence) + 1
        
        return claims
    
    def _find_relevant_context(self, claim: Claim, context: str, max_length: int = 500) -> str:
        """
        Find the most relevant part of context for a claim.
        
        Simple keyword overlap scoring.
        """
        # Split context into chunks
        context_sentences = re.split(r'(?<=[.!?])\s+', context)
        
        # Score each sentence by keyword overlap
        claim_words = set(claim.text.lower().split())
        claim_words -= {'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'ou', 'the', 'a', 'an', 'is', 'are'}
        
        scored_sentences = []
        for sent in context_sentences:
            sent_words = set(sent.lower().split())
            overlap = len(claim_words & sent_words)
            if overlap > 0:
                scored_sentences.append((overlap, sent))
        
        # Sort by overlap and take top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        # Build relevant context
        relevant = []
        total_length = 0
        for score, sent in scored_sentences:
            if total_length + len(sent) > max_length:
                break
            relevant.append(sent)
            total_length += len(sent)
        
        if not relevant:
            # Fallback to first part of context
            return context[:max_length]
        
        return " ".join(relevant)
    
    def _predict_nli(self, premise: str, hypothesis: str) -> Tuple[EntailmentLabel, float]:
        """
        Run NLI prediction on premise-hypothesis pair.
        
        Returns (label, confidence).
        """
        if self.use_mock:
            # Mock: assume entailment for testing
            return EntailmentLabel.ENTAILS, 0.85
        
        if self.model is None:
            return EntailmentLabel.UNKNOWN, 0.0
        
        try:
            # Cross-encoder returns scores for [contradiction, entailment, neutral]
            scores = self.model.predict([(premise, hypothesis)])[0]
            
            # Handle different output formats
            if isinstance(scores, (list, tuple)) and len(scores) == 3:
                contradiction_score, entailment_score, neutral_score = scores
            else:
                # Single score model - treat as entailment probability
                entailment_score = float(scores)
                contradiction_score = 1 - entailment_score
                neutral_score = 0.0
            
            # Determine label
            if entailment_score >= self.entailment_threshold:
                return EntailmentLabel.ENTAILS, float(entailment_score)
            elif contradiction_score >= self.contradiction_threshold:
                return EntailmentLabel.CONTRADICTS, float(contradiction_score)
            elif neutral_score > max(entailment_score, contradiction_score):
                return EntailmentLabel.NEUTRAL, float(neutral_score)
            else:
                return EntailmentLabel.NEUTRAL, float(neutral_score)
                
        except Exception as e:
            logger.error(f"NLI prediction failed: {e}")
            return EntailmentLabel.UNKNOWN, 0.0
    
    def verify_claim(self, claim: Claim, context: str) -> ClaimVerification:
        """
        Verify a single claim against context.
        
        Args:
            claim: The claim to verify
            context: Retrieved context
            
        Returns:
            ClaimVerification with label and score
        """
        # Find relevant context
        relevant_context = self._find_relevant_context(claim, context)
        
        # Run NLI
        label, score = self._predict_nli(relevant_context, claim.text)
        
        # Build explanation
        if label == EntailmentLabel.ENTAILS:
            explanation = f"Claim supported by context (score: {score:.2f})"
        elif label == EntailmentLabel.CONTRADICTS:
            explanation = f"Claim contradicts context (score: {score:.2f})"
        elif label == EntailmentLabel.NEUTRAL:
            explanation = f"Claim not directly supported or contradicted (score: {score:.2f})"
        else:
            explanation = "Could not verify claim"
        
        return ClaimVerification(
            claim=claim,
            label=label,
            score=score,
            supporting_context=relevant_context[:200],
            explanation=explanation
        )
    
    def verify_response(
        self,
        response: str,
        context: str,
        strict: bool = False
    ) -> VerificationResult:
        """
        Verify all claims in a response.
        
        Args:
            response: Generated response text
            context: Retrieved context
            strict: If True, fail on any unverified claim
            
        Returns:
            VerificationResult with overall pass/fail and details
        """
        if not self.enabled:
            return VerificationResult(
                passed=True,
                claims=[],
                overall_score=1.0,
                unverified_claims=[],
                contradicted_claims=[],
                explanation="Verification disabled"
            )
        
        # Extract claims
        claims = self._extract_claims(response)
        
        if not claims:
            return VerificationResult(
                passed=True,
                claims=[],
                overall_score=1.0,
                unverified_claims=[],
                contradicted_claims=[],
                explanation="No verifiable claims found"
            )
        
        # Verify each claim
        verifications = []
        unverified = []
        contradicted = []
        
        for claim in claims:
            verification = self.verify_claim(claim, context)
            verifications.append(verification)
            
            if verification.label == EntailmentLabel.CONTRADICTS:
                contradicted.append(claim)
            elif verification.label in [EntailmentLabel.NEUTRAL, EntailmentLabel.UNKNOWN]:
                unverified.append(claim)
        
        # Calculate overall score
        entailed_count = sum(1 for v in verifications if v.label == EntailmentLabel.ENTAILS)
        overall_score = entailed_count / len(verifications) if verifications else 1.0
        
        # Determine pass/fail
        if contradicted:
            passed = False
            explanation = f"Found {len(contradicted)} contradicted claim(s)"
        elif strict and unverified:
            passed = False
            explanation = f"Found {len(unverified)} unverified claim(s) (strict mode)"
        elif overall_score < 0.5:
            passed = False
            explanation = f"Low verification score: {overall_score:.2f}"
        else:
            passed = True
            explanation = f"Verification passed: {entailed_count}/{len(verifications)} claims supported"
        
        return VerificationResult(
            passed=passed,
            claims=verifications,
            overall_score=overall_score,
            unverified_claims=unverified,
            contradicted_claims=contradicted,
            explanation=explanation
        )
    
    def get_correction_prompt(
        self,
        original_response: str,
        verification_result: VerificationResult
    ) -> str:
        """
        Generate a correction prompt for unsupported claims.
        
        Args:
            original_response: The original response
            verification_result: Verification results
            
        Returns:
            Correction prompt to send to LLM
        """
        unsupported = []
        
        for claim in verification_result.contradicted_claims:
            unsupported.append(f"- CONTRADICTED: {claim.text}")
        
        for claim in verification_result.unverified_claims:
            unsupported.append(f"- UNVERIFIED: {claim.text}")
        
        if not unsupported:
            return ""
        
        prompt = f"""The following claims in your response could not be verified against the provided context:

{chr(10).join(unsupported)}

Please revise your response to:
1. Remove or qualify unverified claims
2. Correct contradicted statements
3. Only state information that is directly supported by the context
4. If information is not available, say "I cannot find this information in the available documents"

Original response:
{original_response}

Revised response:"""
        
        return prompt


class MockNLIVerifier:
    """Mock NLI verifier for testing."""
    
    def __init__(self, **kwargs):
        self.enabled = kwargs.get('enabled', True)
    
    def verify_response(self, response: str, context: str, strict: bool = False) -> VerificationResult:
        """Always pass verification."""
        return VerificationResult(
            passed=True,
            claims=[],
            overall_score=1.0,
            unverified_claims=[],
            contradicted_claims=[],
            explanation="Mock verification - always passes"
        )
    
    def get_correction_prompt(self, original_response: str, verification_result: VerificationResult) -> str:
        return ""


# Singleton instance
_verifier_instance: Optional[NLIVerifier] = None


def get_nli_verifier(
    model_name: str = NLIVerifier.DEFAULT_MODEL,
    entailment_threshold: float = 0.7,
    enabled: bool = True,
    use_mock: bool = False
) -> NLIVerifier:
    """Get or create NLI verifier instance."""
    global _verifier_instance
    if _verifier_instance is None:
        if use_mock:
            _verifier_instance = MockNLIVerifier(enabled=enabled)
        else:
            _verifier_instance = NLIVerifier(
                model_name=model_name,
                entailment_threshold=entailment_threshold,
                enabled=enabled,
                use_mock=use_mock
            )
    return _verifier_instance

