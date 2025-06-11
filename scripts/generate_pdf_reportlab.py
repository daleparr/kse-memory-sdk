#!/usr/bin/env python3
"""
Simple PDF Generator using ReportLab for KSE arXiv Preprint

This uses ReportLab which is more Windows-friendly and doesn't require
external system libraries.
"""

import os
import sys
from pathlib import Path
import re

def install_reportlab():
    """Install ReportLab if not available."""
    try:
        import reportlab
    except ImportError:
        print("Installing reportlab...")
        os.system("pip install reportlab")

def clean_markdown_text(text):
    """Clean markdown text for basic formatting."""
    # Remove markdown headers but keep the text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Convert **bold** to simple text (ReportLab will handle formatting)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Convert *italic* to simple text
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove code blocks but keep content
    text = re.sub(r'```[\w]*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
    
    # Remove inline code formatting
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Remove markdown links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove table formatting
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'^[-\s]+$', '', text, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text.strip()

def generate_pdf_reportlab(markdown_file, output_file):
    """Generate PDF using ReportLab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    except ImportError:
        print("Installing reportlab...")
        os.system("pip install reportlab")
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

    # Read markdown content
    content = Path(markdown_file).read_text(encoding='utf-8')
    
    # Clean the content
    clean_content = clean_markdown_text(content)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_file,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    # Build story
    story = []
    
    # Title page
    story.append(Paragraph("Knowledge Space Embeddings:", title_style))
    story.append(Paragraph("A Hybrid AI Architecture for Scalable Knowledge Retrieval", title_style))
    story.append(Paragraph("with Incremental Learning and Comprehensive Reproducibility", title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("[To be filled]", styles['Normal']))
    story.append(Paragraph("[Affiliation to be filled]", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("December 2025", styles['Normal']))
    story.append(Paragraph("Version 3.0", styles['Normal']))
    story.append(Paragraph("arXiv Preprint", styles['Normal']))
    story.append(PageBreak())
    
    # Split content into sections
    sections = clean_content.split('\n\n')
    
    for section in sections:
        if not section.strip():
            continue
            
        # Check if this looks like a heading (starts with capital and is short)
        if len(section) < 100 and section[0].isupper() and not section.endswith('.'):
            story.append(Paragraph(section, heading_style))
        else:
            # Split long paragraphs
            paragraphs = section.split('\n')
            for para in paragraphs:
                if para.strip():
                    story.append(Paragraph(para, body_style))
                    story.append(Spacer(1, 6))
    
    # Build PDF
    try:
        doc.build(story)
        print(f"PDF generated successfully: {output_file}")
        return True
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False

def main():
    """Main function."""
    print("KSE arXiv PDF Generator (ReportLab)")
    print("=" * 40)
    
    markdown_file = "KSE_ARXIV_PREPRINT_V3.md"
    output_file = "KSE_ARXIV_PREPRINT_V3_reportlab.pdf"
    
    if not Path(markdown_file).exists():
        print(f"Markdown file not found: {markdown_file}")
        sys.exit(1)
    
    print("Installing dependencies...")
    install_reportlab()
    
    print("Converting markdown to PDF...")
    if generate_pdf_reportlab(markdown_file, output_file):
        print(f"\nPDF generated successfully!")
        print(f"Output: {output_file}")
        print("\nNote: This is a basic conversion. For better formatting,")
        print("consider using pandoc with LaTeX if available.")
    else:
        print("\nPDF generation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()