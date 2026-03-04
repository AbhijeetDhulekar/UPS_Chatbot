"""
Chat history export to PDF
Uses reportlab to create a clean PDF report.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import io
from typing import List, Dict

def generate_pdf_report(session_id: str, messages: List[Dict]) -> bytes:
    """
    Generate a PDF report of the chat history.
    
    Args:
        session_id: Unique session identifier
        messages: List of message dicts with 'role', 'content', 'timestamp'
    
    Returns:
        PDF as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    # Custom styles
    title_style = styles['Heading1']
    normal_style = styles['Normal']
    user_style = ParagraphStyle(
        'UserStyle',
        parent=styles['Normal'],
        textColor=colors.darkblue,
        leftIndent=20,
        spaceAfter=10
    )
    assistant_style = ParagraphStyle(
        'AssistantStyle',
        parent=styles['Normal'],
        textColor=colors.darkgreen,
        leftIndent=20,
        spaceAfter=10
    )
    
    story = []
    
    # Title
    story.append(Paragraph(f"Chat Report - Session {session_id}", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Messages
    for msg in messages:
        role = msg['role'].capitalize()
        content = msg['content']
        timestamp = msg.get('timestamp', '')
        
        # Style based on role
        if role == 'User':
            pstyle = user_style
        else:
            pstyle = assistant_style
        
        # Format message
        text = f"<b>{role} ({timestamp}):</b><br/>{content}"
        story.append(Paragraph(text, pstyle))
        story.append(Spacer(1, 0.1*inch))
    
    # Add sources if present (optional)
    # Could include a table of sources for each answer
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes