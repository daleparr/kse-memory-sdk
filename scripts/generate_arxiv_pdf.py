#!/usr/bin/env python3
"""
KSE arXiv Preprint PDF Generator

This script converts the KSE_ARXIV_PREPRINT_V3.md to a properly formatted academic PDF
using pandoc with LaTeX for high-quality academic publication formatting.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    dependencies = ['pandoc', 'pdflatex']
    missing = []
    
    for dep in dependencies:
        if not shutil.which(dep):
            missing.append(dep)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("\nTo install:")
        print("- pandoc: https://pandoc.org/installing.html")
        print("- pdflatex: Install TeX Live or MiKTeX")
        return False
    
    print("✅ All dependencies found")
    return True

def create_latex_template():
    """Create a custom LaTeX template for academic formatting."""
    template_content = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{geometry}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}
\usepackage{wrapfig}
\usepackage{float}
\usepackage{colortbl}
\usepackage{pdflscape}
\usepackage{tabu}
\usepackage{threeparttable}
\usepackage{threeparttablex}
\usepackage[normalem]{ulem}
\usepackage{makecell}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{cleveref}
\usepackage{natbib}

% Page geometry
\geometry{
    a4paper,
    left=2.5cm,
    right=2.5cm,
    top=2.5cm,
    bottom=2.5cm
}

% Headers and footers
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{Knowledge Space Embeddings}
\fancyhead[R]{arXiv Preprint v3.0}
\fancyfoot[C]{\thepage}

% Code listings
\lstset{
    basicstyle=\ttfamily\footnotesize,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10},
    numbers=left,
    numberstyle=\tiny,
    stepnumber=1,
    showstringspaces=false
}

% Hyperref setup
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,
    urlcolor=cyan,
    citecolor=blue,
    pdftitle={Knowledge Space Embeddings: A Hybrid AI Architecture},
    pdfauthor={KSE Research Team},
    pdfsubject={Artificial Intelligence, Knowledge Retrieval},
    pdfkeywords={Knowledge Retrieval, Hybrid AI, Incremental Learning}
}

% Title formatting
\title{\LARGE\textbf{Knowledge Space Embeddings: A Hybrid AI Architecture for Scalable Knowledge Retrieval with Incremental Learning and Comprehensive Reproducibility}}
\author{[To be filled]\\[0.5em]\textit{[Affiliation to be filled]}}
\date{December 2025 \\ Version 3.0 \\ arXiv Preprint}

\begin{document}

\maketitle

\begin{abstract}
$abstract$
\end{abstract}

\textbf{Keywords:} $keywords$

\newpage
\tableofcontents
\newpage

$body$

\end{document}
"""
    
    template_path = Path("kse-memory-sdk/scripts/arxiv_template.tex")
    template_path.write_text(template_content.strip())
    return template_path

def preprocess_markdown():
    """Preprocess the markdown file for better PDF conversion."""
    input_file = Path("kse-memory-sdk/KSE_ARXIV_PREPRINT_V3.md")
    output_file = Path("kse-memory-sdk/scripts/KSE_ARXIV_PREPRINT_V3_processed.md")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return None
    
    content = input_file.read_text(encoding='utf-8')
    
    # Extract abstract and keywords for template
    abstract_start = content.find("## Abstract")
    abstract_end = content.find("**Keywords:**")
    keywords_end = content.find("\n\n", abstract_end)
    
    if abstract_start != -1 and abstract_end != -1:
        abstract = content[abstract_start + len("## Abstract"):abstract_end].strip()
        keywords = content[abstract_end + len("**Keywords:**"):keywords_end].strip()
        
        # Remove the abstract and keywords from main content since they'll be in template
        content = content[:abstract_start] + content[keywords_end:]
    else:
        abstract = "Abstract not found"
        keywords = "Keywords not found"
    
    # Fix code blocks for better LaTeX rendering
    content = content.replace("```yaml", "```{.yaml}")
    content = content.replace("```python", "```{.python}")
    content = content.replace("```json", "```{.json}")
    
    # Fix table formatting
    content = content.replace("| **", "| \\textbf{")
    content = content.replace("** |", "} |")
    
    # Save processed content
    output_file.write_text(content, encoding='utf-8')
    
    return output_file, abstract, keywords

def generate_pdf():
    """Generate the PDF using pandoc with LaTeX."""
    print("🔄 Preprocessing markdown...")
    result = preprocess_markdown()
    if not result:
        return False
    
    processed_file, abstract, keywords = result
    
    print("🔄 Creating LaTeX template...")
    template_path = create_latex_template()
    
    print("🔄 Converting to PDF...")
    
    # Pandoc command with academic formatting
    cmd = [
        'pandoc',
        str(processed_file),
        '--template', str(template_path),
        '--pdf-engine=pdflatex',
        '--variable', f'abstract={abstract}',
        '--variable', f'keywords={keywords}',
        '--number-sections',
        '--toc',
        '--toc-depth=3',
        '--listings',
        '--highlight-style=tango',
        '--bibliography=references.bib',  # If bibliography exists
        '--csl=ieee.csl',  # Citation style
        '-o', 'kse-memory-sdk/KSE_ARXIV_PREPRINT_V3.pdf'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ PDF generated successfully!")
            print(f"📄 Output: kse-memory-sdk/KSE_ARXIV_PREPRINT_V3.pdf")
            
            # Clean up temporary files
            processed_file.unlink(missing_ok=True)
            template_path.unlink(missing_ok=True)
            
            return True
        else:
            print(f"❌ PDF generation failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ pandoc not found. Please install pandoc first.")
        return False
    except Exception as e:
        print(f"❌ Error during PDF generation: {e}")
        return False

def main():
    """Main function to generate the arXiv PDF."""
    print("🚀 KSE arXiv Preprint PDF Generator")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Generate PDF
    if generate_pdf():
        print("\n✅ PDF generation completed successfully!")
        print("📄 File: kse-memory-sdk/KSE_ARXIV_PREPRINT_V3.pdf")
        print("\n📋 Next steps:")
        print("1. Review the generated PDF for formatting")
        print("2. Add author information and affiliations")
        print("3. Submit to arXiv or academic venue")
    else:
        print("\n❌ PDF generation failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()