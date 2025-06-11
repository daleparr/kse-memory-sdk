@echo off
echo ========================================
echo KSE arXiv Preprint PDF Generator
echo ========================================
echo.

echo Checking for pandoc and LaTeX...
where pandoc >nul 2>&1
if %errorlevel% == 0 (
    where pdflatex >nul 2>&1
    if %errorlevel% == 0 (
        echo Found pandoc and LaTeX - using high-quality generator
        python scripts/generate_arxiv_pdf.py
        goto :end
    )
)

echo Using ReportLab Python generator (reliable, Windows-friendly)
python scripts/generate_pdf_reportlab.py

:end
echo.
echo PDF generation complete!
echo Check the generated PDF file in the current directory.
pause