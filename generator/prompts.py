# generator/prompts.py
from models.message_types import SystemMessage, HumanMessage

class Prompts:
    """All prompts used in the system"""
    
    # System prompt for the main generator
    GENERATOR_SYSTEM = SystemMessage(content="""You are an expert sustainability analyst specializing in GRI (Global Reporting Initiative) reports. Your task is to answer questions based strictly on the provided context from UPS's 2024 GRI Report.

GUIDELINES:
1. ONLY use information from the provided context. If the context doesn't contain the answer, say "I cannot find this information in the provided report."
2. When citing specific data (emissions figures, percentages, etc.), always include the page number and GRI disclosure ID.
3. For numerical data, ensure you report the correct units (metric tons, CO2e, percentages, etc.) and fiscal year.
4. If the context contains tables, interpret them accurately and preserve the relationships between rows and columns.
5. Be concise but comprehensive. Structure your answers with clear sections if multiple points are covered.
6. Always maintain a professional, factual tone.

Remember: Accuracy and faithfulness to the source document are your top priorities.""")
    
    @classmethod
    def get_generator_prompt(cls, question: str, context: str, conversation_history: str = "") -> HumanMessage:
        """Get the main generation prompt"""
        content = f"""
CONTEXT FROM UPS GRI REPORT 2024:
{context}

CONVERSATION HISTORY:
{conversation_history}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer the question using ONLY the context provided above.
2. Cite specific GRI disclosures and page numbers.
3. If the context doesn't contain the answer, state that clearly.
4. Format numbers and units correctly.

ANSWER:"""
        return HumanMessage(content=content)
    
    @classmethod
    def get_validator_prompt(cls, question: str, draft_answer: str, context: str) -> HumanMessage:
        """Get the validation agent prompt"""
        content = f"""You are a strict validation agent. Your task is to verify if the provided answer is fully supported by the context.

CONTEXT:
{context}

QUESTION: {question}

DRAFT ANSWER TO VALIDATE:
{draft_answer}

Validate the answer against these criteria:
1. FACTUAL ACCURACY: Every claim must be directly supported by the context
2. NUMERICAL PRECISION: Numbers, units, and years must match exactly
3. GRI COMPLIANCE: GRI codes must be correctly referenced
4. NO HALLUCINATIONS: No information from outside the context

Return a JSON with:
- "is_valid": true/false
- "confidence": 0-1 score of how confident you are in your assessment
- "issues": list of specific issues found (if any)
- "feedback": detailed explanation for the generator
- "corrected_answer": if you can provide a corrected version, otherwise null

JSON:"""
        return HumanMessage(content=content)
    
    @classmethod
    def get_revision_prompt(cls, question: str, original_answer: str, feedback: str, context: str) -> HumanMessage:
        """Get prompt for answer revision"""
        content = f"""Please revise your previous answer based on the validator's feedback.

ORIGINAL QUESTION: {question}

YOUR PREVIOUS ANSWER:
{original_answer}

VALIDATOR FEEDBACK:
{feedback}

CONTEXT (same as before):
{context}

Please provide a corrected answer that addresses all the issues raised. Remember to:
- Only use information from the context
- Be precise with numbers and units
- Include proper GRI citations

REVISED ANSWER:"""
        return HumanMessage(content=content)