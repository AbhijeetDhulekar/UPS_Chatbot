"""
Evaluation metrics for RAG system
Computes precision, recall, faithfulness, etc.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from langchain_core.documents import Document
from config import Config
from debug.debugger import debugger

def evaluate_answer(question: str, answer: str, context: str, ground_truth: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate a single Q&A pair using Ragas metrics.
    If ground_truth is provided, also compute exact match and F1.
    """
    try:
        # Prepare data in Ragas format
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [[context]]  # List of lists
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]
        
        dataset = Dataset.from_dict(data)
        
        # Select metrics
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        
        result = evaluate(dataset, metrics=metrics)
        
        # Convert to dict
        scores = result.to_pandas().iloc[0].to_dict()
        
        # Compute additional metrics if ground truth available
        if ground_truth:
            # Exact match
            exact_match = 1.0 if answer.strip().lower() == ground_truth.strip().lower() else 0.0
            scores["exact_match"] = exact_match
            
            # Token overlap F1 (simple)
            answer_tokens = set(answer.lower().split())
            truth_tokens = set(ground_truth.lower().split())
            if truth_tokens:
                precision = len(answer_tokens & truth_tokens) / len(answer_tokens) if answer_tokens else 0
                recall = len(answer_tokens & truth_tokens) / len(truth_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                scores["f1"] = f1
        
        debugger.log("EVALUATION", scores)
        return scores
        
    except Exception as e:
        debugger.log("EVALUATION_ERROR", str(e), level="ERROR")
        return {}

def evaluate_retrieval(queries: List[str], relevant_docs: List[List[str]], retrieved_docs: List[List[str]], k_values: List[int] = [1,3,5]):
    """
    Compute precision@k and recall@k for retrieval.
    """
    metrics = {}
    for k in k_values:
        precisions = []
        recalls = []
        for q_idx in range(len(queries)):
            retrieved = retrieved_docs[q_idx][:k]
            relevant = set(relevant_docs[q_idx])
            if not relevant:
                continue
            retrieved_set = set(retrieved)
            tp = len(relevant & retrieved_set)
            precisions.append(tp / k)
            recalls.append(tp / len(relevant))
        metrics[f"precision@{k}"] = np.mean(precisions) if precisions else 0
        metrics[f"recall@{k}"] = np.mean(recalls) if recalls else 0
    return metrics