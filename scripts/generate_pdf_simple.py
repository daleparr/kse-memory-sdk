#!/usr/bin/env python3
"""
Simple PDF Generator for KSE arXiv Preprint

Alternative PDF generation using Python libraries (markdown2, weasyprint)
for systems without pandoc/LaTeX.
"""

import os
import sys
from pathlib import Path

def install_dependencies():
    """Install required Python packages."""
    packages = ['markdown2', 'weasyprint', 'pygments']
    
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")

def create_html_template():
    """Create HTML template for PDF conversion."""
    template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Knowledge Space Embeddings - arXiv Preprint v3.0</title>
    <style>
        @page {
            size: A4;
            margin: 2.5cm;
            @top-left { content: "Knowledge Space Embeddings"; }
            @top-right { content: "arXiv Preprint v3.0"; }
            @bottom-center { content: counter(page); }
        }
        
        body {
            font-family: 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            max-width: none;
        }
        
        h1 {
            font-size: 18pt;
            font-weight: bold;
            text-align: center;
            margin: 2em 0 1em 0;
            page-break-before: auto;
        }
        
        h2 {
            font-size: 14pt;
            font-weight: bold;
            margin: 1.5em 0 0.5em 0;
            border-bottom: 1px solid #ccc;
            padding-bottom: 0.2em;
        }
        
        h3 {
            font-size: 12pt;
            font-weight: bold;
            margin: 1em 0 0.5em 0;
        }
        
        h4 {
            font-size: 11pt;
            font-weight: bold;
            margin: 0.8em 0 0.3em 0;
        }
        
        p {
            margin: 0.5em 0;
            text-align: justify;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
            font-size: 10pt;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 0.5em;
            text-align: left;
        }
        
        th {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        
        code {
            font-family: 'Courier New', monospace;
            background-color: #f5f5f5;
            padding: 0.1em 0.3em;
            border-radius: 3px;
            font-size: 10pt;
        }
        
        pre {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 1em;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.4;
        }
        
        pre code {
            background: none;
            padding: 0;
        }
        
        .title-page {
            text-align: center;
            page-break-after: always;
        }
        
        .title {
            font-size: 20pt;
            font-weight: bold;
            margin: 3em 0 2em 0;
            line-height: 1.3;
        }
        
        .authors {
            font-size: 14pt;
            margin: 2em 0;
        }
        
        .date {
            font-size: 12pt;
            margin: 1em 0;
        }
        
        .abstract {
            margin: 2em 0;
            padding: 1em;
            background-color: #f9f9f9;
            border-left: 4px solid #007bff;
        }
        
        .keywords {
            font-weight: bold;
            margin: 1em 0;
        }
        
        .toc {
            page-break-after: always;
        }
        
        .toc h2 {
            border-bottom: none;
        }
        
        .toc ul {
            list-style: none;
            padding-left: 0;
        }
        
        .toc li {
            margin: 0.3em 0;
        }
        
        .page-break {
            page-break-before: always;
        }
        
        blockquote {
            margin: 1em 2em;
            padding: 0.5em 1em;
            border-left: 3px solid #ccc;
            background-color: #f9f9f9;
        }
        
        ul, ol {
            margin: 0.5em 0;
            padding-left: 2em;
        }
        
        li {
            margin: 0.2em 0;
        }
        
        strong {
            font-weight: bold;
        }
        
        em {
            font-style: italic;
        }
        
        .equation {
            text-align: center;
            margin: 1em 0;
            font-style: italic;
        }
        
        .figure {
            text-align: center;
            margin: 1em 0;
        }
        
        .figure-caption {
            font-size: 10pt;
            font-style: italic;
            margin-top: 0.5em;
        }
    </style>
</head>
<body>
    <div class="title-page">
        <div class="title">
            Knowledge Space Embeddings: A Hybrid AI Architecture for Scalable Knowledge Retrieval with Incremental Learning and Comprehensive Reproducibility
        </div>
        <div class="authors">
            [To be filled]<br>
            <em>[Affiliation to be filled]</em>
        </div>
        <div class="date">
            December 2025<br>
            Version 3.0<br>
            arXiv Preprint
        </div>
    </div>
    
    {content}
</body>
</html>
"""
    return template

def convert_markdown_to_html(markdown_file):
    """Convert markdown to HTML."""
    try:
        import markdown2
    except ImportError:
        print("Installing markdown2...")
        os.system("pip install markdown2")
        import markdown2
    
    # Read markdown content
    content = Path(markdown_file).read_text(encoding='utf-8')
    
    # Extract abstract and keywords
    abstract_start = content.find("## Abstract")
    abstract_end = content.find("**Keywords:**")
    keywords_end = content.find("\n\n", abstract_end)
    
    abstract_content = ""
    keywords_content = ""
    
    if abstract_start != -1 and abstract_end != -1:
        abstract_text = content[abstract_start + len("## Abstract"):abstract_end].strip()
        keywords_text = content[abstract_end + len("**Keywords:**"):keywords_end].strip()
        
        abstract_content = f'<div class="abstract"><h2>Abstract</h2>{markdown2.markdown(abstract_text)}</div>'
        keywords_content = f'<div class="keywords"><strong>Keywords:</strong> {keywords_text}</div>'
        
        # Remove abstract and keywords from main content
        content = content[:abstract_start] + content[keywords_end:]
    
    # Convert main content
    html_content = markdown2.markdown(
        content,
        extras=[
            'fenced-code-blocks',
            'tables',
            'header-ids',
            'toc',
            'code-friendly',
            'footnotes'
        ]
    )
    
    # Add abstract and keywords back
    full_content = abstract_content + keywords_content + '<div class="page-break"></div>' + html_content
    
    # Create full HTML
    template = create_html_template()
    html = template.format(content=full_content)
    
    return html

def generate_pdf_simple(markdown_file, output_file):
    """Generate PDF using weasyprint."""
    try:
        from weasyprint import HTML, CSS
    except ImportError:
        print("Installing weasyprint...")
        os.system("pip install weasyprint")
        from weasyprint import HTML, CSS
    
    print("Converting markdown to HTML...")
    html_content = convert_markdown_to_html(markdown_file)
    
    # Save HTML for debugging
    html_file = Path(output_file).with_suffix('.html')
    html_file.write_text(html_content, encoding='utf-8')
    
    print("Converting HTML to PDF...")
    try:
        HTML(string=html_content).write_pdf(output_file)
        print(f"PDF generated: {output_file}")
        
        # Clean up HTML file
        html_file.unlink(missing_ok=True)
        
        return True
    except Exception as e:
        print(f"PDF generation failed: {e}")
        print(f"HTML file saved for debugging: {html_file}")
        return False

def main():
    """Main function."""
    print("Simple KSE arXiv PDF Generator")
    print("=" * 40)
    
    markdown_file = "KSE_ARXIV_PREPRINT_V3.md"
    output_file = "KSE_ARXIV_PREPRINT_V3_simple.pdf"
    
    if not Path(markdown_file).exists():
        print(f"Markdown file not found: {markdown_file}")
        sys.exit(1)
    
    print("Installing dependencies...")
    install_dependencies()
    
    if generate_pdf_simple(markdown_file, output_file):
        print(f"\nPDF generated successfully!")
        print(f"Output: {output_file}")
    else:
        print("\nPDF generation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()