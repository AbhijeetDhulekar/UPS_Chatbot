# ingestion/parser.py
import pdfplumber
import fitz  # PyMuPDF
from typing import List, Dict, Any
from loguru import logger
from debug.debugger import debugger

class PDFParser:
    """Layout-aware PDF parser"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
    
    @debugger.timer
    def parse(self) -> tuple[List[Dict], List[Dict]]:
        """
        Parse PDF and extract text blocks and tables
        Returns: (text_blocks, tables)
        """
        text_blocks = []
        tables = []
        
        try:
            # Extract text with PyMuPDF (structure-aware)
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    # Detect headers by font size/style
                                    is_header = (
                                        span["size"] > 12 or 
                                        "bold" in span["font"].lower() or
                                        any(keyword in text for keyword in ["GRI", "Disclosure", "Report"])
                                    )
                                    
                                    text_blocks.append({
                                        "text": text,
                                        "page": page_num + 1,
                                        "is_header": is_header,
                                        "font_size": span["size"],
                                        "font": span["font"],
                                        "bbox": span["bbox"]
                                    })
            
            # Extract tables with pdfplumber
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for table_data in page_tables:
                        if table_data:  # Non-empty table
                            tables.append({
                                "page": page_num + 1,
                                "data": table_data,
                                "markdown": self._table_to_markdown(table_data)
                            })
            
            debugger.log("PARSING", {
                "text_blocks": len(text_blocks),
                "tables": len(tables),
                "pages": len(self.doc)
            })
            
            return text_blocks, tables
            
        except Exception as e:
            debugger.log("PARSING", str(e), level="ERROR")
            raise
    
    def _table_to_markdown(self, table_data: List[List]) -> str:
        """Convert table to markdown format"""
        if not table_data:
            return ""
        
        # Create header
        header = table_data[0]
        markdown = "| " + " | ".join([str(cell or "") for cell in header]) + " |\n"
        markdown += "|" + "|".join(["---" for _ in header]) + "|\n"
        
        # Add rows
        for row in table_data[1:]:
            markdown += "| " + " | ".join([str(cell or "") for cell in row]) + " |\n"
        
        return markdown