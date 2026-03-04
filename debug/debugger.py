# debug/debugger.py
"""
Enhanced Debugger for RAG System
Tracks all components and helps identify issues
"""

from loguru import logger
import time
import json
import traceback
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import sys
import os

class RAGDebugger:
    """Comprehensive debugging system for RAG pipeline"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.debug_log = []
        self.errors = []
        self.warnings = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.log_file = log_dir / f"debug_session_{self.session_id}.json"
        self.error_log = log_dir / f"errors_{self.session_id}.log"
        
        # Component status tracking
        self.component_status = {
            "vector_store": "unknown",
            "hybrid_search": "unknown",
            "reranker": "unknown",
            "llm_client": "unknown",
            "validator": "unknown",
            "memory": "unknown",
            "cache": "unknown"
        }
        
        # Performance tracking
        self.performance_stats = {}
        
        logger.add(
            self.error_log,
            rotation="10 MB",
            retention="1 week",
            level="ERROR"
        )
        
        self.log("DEBUGGER_INIT", {"session_id": self.session_id})
    
    def log(self, stage: str, data: Any, level: str = "INFO"):
        """Log debug information"""
        if not self.enabled:
            return
            
        entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "level": level,
            "data": data
        }
        
        self.debug_log.append(entry)
        
        # Log to file
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except:
            pass
        
        # Log to console with appropriate level
        if level == "ERROR":
            logger.error(f"[{stage}] {data}")
            self.errors.append(entry)
        elif level == "WARNING":
            logger.warning(f"[{stage}] {data}")
            self.warnings.append(entry)
        else:
            logger.info(f"[{stage}] {data}")
    
    def log_error(self, stage: str, error: Exception, context: Dict = None):
        """Log error with full traceback"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        self.log(stage, error_data, level="ERROR")
    
    def timer(self, func: Callable) -> Callable:
        """Decorator to time functions"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)
            
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                
                self.log(
                    stage=f"PERFORMANCE",
                    data={
                        "function": func.__name__,
                        "module": func.__module__,
                        "duration_seconds": round(duration, 3),
                        "status": "success"
                    },
                    level="INFO"
                )
                
                # Store performance stats
                if func.__name__ not in self.performance_stats:
                    self.performance_stats[func.__name__] = []
                self.performance_stats[func.__name__].append(duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start
                self.log_error(func.__name__, e, {
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200],
                    "duration": duration
                })
                raise
        
        return wrapper
    
    def check_component(self, component_name: str, component: Any) -> bool:
        """Check if a component is properly initialized"""
        try:
            if component is None:
                self.component_status[component_name] = "missing"
                self.log("COMPONENT_CHECK", {
                    "component": component_name,
                    "status": "missing"
                }, level="WARNING")
                return False
            
            # Basic health check
            if hasattr(component, '__class__'):
                self.component_status[component_name] = "loaded"
                self.log("COMPONENT_CHECK", {
                    "component": component_name,
                    "type": component.__class__.__name__,
                    "status": "loaded"
                })
                return True
            else:
                self.component_status[component_name] = "invalid"
                return False
                
        except Exception as e:
            self.component_status[component_name] = "error"
            self.log_error(f"COMPONENT_CHECK_{component_name}", e)
            return False
    
    def trace_retrieval(self, query: str, results: list):
        """Trace retrieval process"""
        self.log(
            stage="RETRIEVAL",
            data={
                "query": query[:100],
                "num_results": len(results),
                "top_scores": [r.get("score", 0) for r in results[:3]] if results else []
            }
        )
    
    def trace_generation(self, prompt: str, response: str, validation: dict = None):
        """Trace generation process"""
        self.log(
            stage="GENERATION",
            data={
                "prompt_preview": prompt[:200],
                "response_preview": response[:200],
                "validation": validation,
                "response_length": len(response)
            }
        )
    
    def trace_validation(self, validation_result: dict):
        """Trace validation process"""
        self.log(
            stage="VALIDATION",
            data=validation_result,
            level="WARNING" if not validation_result.get("is_valid") else "INFO"
        )
    
    def get_system_report(self) -> Dict:
        """Get comprehensive system status report"""
        return {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "component_status": self.component_status,
            "total_logs": len(self.debug_log),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "performance": {
                name: {
                    "avg": sum(times) / len(times) if times else 0,
                    "min": min(times) if times else 0,
                    "max": max(times) if times else 0,
                    "count": len(times)
                }
                for name, times in self.performance_stats.items()
            },
            "log_file": str(self.log_file),
            "error_log": str(self.error_log)
        }
    
    def get_summary(self) -> dict:
        """Get debug summary (simplified version)"""
        stages = {}
        for entry in self.debug_log:
            stage = entry["stage"]
            if stage not in stages:
                stages[stage] = {"count": 0, "errors": 0, "warnings": 0}
            stages[stage]["count"] += 1
            if entry["level"] == "ERROR":
                stages[stage]["errors"] += 1
            elif entry["level"] == "WARNING":
                stages[stage]["warnings"] += 1
        
        return {
            "session_id": self.session_id,
            "total_entries": len(self.debug_log),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "stages": stages,
            "component_status": self.component_status,
            "log_file": str(self.log_file)
        }
    
    def print_report(self):
        """Print formatted system report"""
        report = self.get_system_report()
        
        print("\n" + "="*60)
        print(" RAG SYSTEM DEBUG REPORT")
        print("="*60)
        print(f"Session ID: {report['session_id']}")
        print(f"Timestamp: {report['timestamp']}")
        print("-"*60)
        
        print("\n🔧 COMPONENT STATUS:")
        for comp, status in report['component_status'].items():
            icon = "" if status == "loaded" else "!!!" if status == "missing" else "XXX"
            print(f"  {icon} {comp}: {status}")
        
        print("\n PERFORMANCE STATS:")
        for func, stats in report['performance'].items():
            print(f"  {func}: avg={stats['avg']:.2f}s, count={stats['count']}")
        
        print("\n  ERRORS & WARNINGS:")
        print(f"  Errors: {report['errors']}")
        print(f"  Warnings: {report['warnings']}")
        
        print("\n LOG FILES:")
        print(f"  Debug log: {report['log_file']}")
        print(f"  Error log: {report['error_log']}")
        print("="*60)

# Global debugger instance
debugger = RAGDebugger(enabled=True)

# Function to check entire system
def check_system():
    """Quick system health check"""
    print("\n Running system health check...")
    
    # Check critical directories
    dirs = ['data', 'chroma_db/faiss', 'logs']
    for d in dirs:
        path = Path(d)
        if path.exists():
            print(f" {d}/ exists")
        else:
            print(f" {d}/ missing")
            path.mkdir(parents=True, exist_ok=True)
            print(f"   Created {d}/")
    
    # Check PDF
    pdf_path = Path('data/2024-UPS-GRI-Report.pdf')
    if pdf_path.exists():
        size = pdf_path.stat().st_size / (1024 * 1024)
        print(f" PDF found ({size:.1f} MB)")
    else:
        print(f" PDF not found at {pdf_path}")
    
    # Check FAISS index
    faiss_path = Path('chroma_db/faiss')
    index_file = faiss_path / 'index.faiss'
    data_file = faiss_path / 'data.pkl'
    
    if index_file.exists() and data_file.exists():
        print(f" FAISS index found")
    else:
        print(f" FAISS index incomplete")
    
    print(" Health check complete!\n")