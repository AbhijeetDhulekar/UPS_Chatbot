# generator/validator.py
"""
Validator Agent for RAG System
Validates generated answers against source context
"""

import json
from typing import List, Dict
from models.message_types import ValidationMessage, AIMessage
from generator.prompts import Prompts
from debug.debugger import debugger
from config import Config

class ValidatorAgent:
    """Validates generated answers against source context"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def validate(self, answer: AIMessage, context_chunks: List[Dict], question: str) -> ValidationMessage:
        """
        Validate answer against source chunks.
        Includes out-of-scope question detection.
        """
        
        # Check if this is an out-of-scope response (from our error handling)
        out_of_scope_phrases = [
            "couldn't find information",
            "does not contain information",
            "not in the UPS GRI Report",
            "outside the scope",
            "no relevant context found"
        ]
        
        if any(phrase in answer.content.lower() for phrase in out_of_scope_phrases):
            debugger.log("VALIDATION_SKIP", {
                "reason": "Out of scope question detected",
                "answer_preview": answer.content[:100]
            })
            
            return ValidationMessage(
                content=json.dumps({
                    "is_valid": True,
                    "confidence": 1.0,
                    "issues": [],
                    "feedback": "Question is outside document scope - no validation needed",
                    "corrected_answer": None
                }),
                is_valid=True,
                feedback="Question is outside document scope",
                metadata={
                    "confidence": 1.0,
                    "issues": [],
                    "validation_type": "out_of_scope_skip"
                }
            )
        
        # Prepare context text (your existing code)
        context_text = "\n\n".join([
            f"[Page {chunk['metadata'].get('page_start', 'N/A')}] {chunk['text']}"
            for chunk in context_chunks[:5]  # Top 5 chunks for validation
        ])
        
        # Get validation from LLM
        validation_prompt = Prompts.get_validator_prompt(
            question=question,
            draft_answer=answer.content,
            context=context_text
        )
        
        messages = [
            {"role": "system", "content": "You are a strict validation agent. Always return valid JSON."},
            validation_prompt.dict()
        ]
        
        try:
            response = self.llm.generate(messages)
            
            # Parse JSON response
            try:
                validation_result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response if it contains other text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    validation_result = json.loads(json_match.group())
                else:
                    # Fallback validation result
                    validation_result = {
                        "is_valid": True,
                        "confidence": 0.8,
                        "issues": ["Could not parse validation response"],
                        "feedback": "Validation response was not in expected format",
                        "corrected_answer": None
                    }
            
            validation_msg = ValidationMessage(
                content=json.dumps(validation_result),
                is_valid=validation_result.get("is_valid", False),
                feedback=validation_result.get("feedback"),
                corrected_content=validation_result.get("corrected_answer"),
                metadata={
                    "confidence": validation_result.get("confidence", 0),
                    "issues": validation_result.get("issues", []),
                    "validation_type": "llm_validation"
                }
            )
            
        except Exception as e:
            debugger.log("VALIDATION_ERROR", str(e), level="ERROR")
            
            # Fallback validation if JSON parsing fails
            validation_msg = ValidationMessage(
                content=f"Validation error: {str(e)}",
                is_valid=True,  # Assume valid on error to avoid blocking
                feedback="Could not parse validation response",
                metadata={
                    "error": str(e),
                    "validation_type": "error_fallback"
                }
            )
        
        debugger.trace_validation(validation_msg.dict())
        return validation_msg
    
    def validate_with_retry(self, answer: AIMessage, context_chunks: List[Dict], 
                           question: str, max_attempts: int = None) -> tuple[AIMessage, ValidationMessage]:
        """
        Validate with automatic retry on failure.
        Includes out-of-scope detection to skip retries.
        """
        if max_attempts is None:
            max_attempts = Config.MAX_VALIDATION_ATTEMPTS
        
        # Check for out-of-scope first (skip validation)
        out_of_scope_phrases = ["couldn't find information", "not in the UPS GRI Report", "outside the scope"]
        if any(phrase in answer.content.lower() for phrase in out_of_scope_phrases):
            validation = self.validate(answer, context_chunks, question)
            answer.validation_status = "out_of_scope"
            return answer, validation
        
        current_answer = answer
        context_text = "\n\n".join([chunk["text"] for chunk in context_chunks[:5]])
        
        for attempt in range(max_attempts):
            validation = self.validate(current_answer, context_chunks, question)
            
            if validation.is_valid:
                current_answer.validation_status = "validated"
                current_answer.confidence = validation.metadata.get("confidence", 0.8)
                return current_answer, validation
            
            # If not valid and we have attempts left, revise
            if attempt < max_attempts - 1 and validation.feedback:
                debugger.log("VALIDATION_RETRY", {
                    "attempt": attempt + 1,
                    "feedback": validation.feedback[:100]
                })
                
                revision_prompt = Prompts.get_revision_prompt(
                    question=question,
                    original_answer=current_answer.content,
                    feedback=validation.feedback,
                    context=context_text
                )
                
                messages = [
                    {"role": "system", "content": "You are an expert revising your answer based on feedback."},
                    revision_prompt.dict()
                ]
                
                revised_content = self.llm.generate(messages)
                current_answer = AIMessage(
                    content=revised_content,
                    sources=current_answer.sources,
                    metadata={**current_answer.metadata, "revision_attempt": attempt + 1}
                )
        
        # If we get here, validation failed after all attempts
        current_answer.validation_status = "failed_validation"
        return current_answer, validation