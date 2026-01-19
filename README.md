# OCR-Handwriting-to-Text-Question-Answer-Segmentation-without-LLM
##  Overview

This project implements an automated system to extract and separate questions and answers from handwritten documents using OCR + heuristic-based parsing, without relying on any Large Language Models (LLMs).

The system converts handwritten text into structured digital output while handling:
- Multi-line questions and answers
- Subdivisions like a), b), c)
- Cross-image continuity
- Missing explicit markers

## Features

- Handwritten text recognition using Tesseract OCR
- Image preprocessing optimized for handwriting
- Priority-based question–answer classification
- Subdivision detection (a), b), c))
- Multi-image document support
- Robust handling of missing markers
- Clean formatted output

## How It Works
System Architecture
`Input Image
→ Image Preprocessing
→ OCR Text Extraction
→ Line-by-Line Parsing
→ Priority-Based Classification
→ Structured Q&A Output`

## Technologies Used
- Python 3.x
- OpenCV
- Tesseract OCR
- Regex (pattern matching)
- NumPy

## Limitations
- OCR accuracy depends on handwriting quality
- Nested subdivisions not fully hierarchical
- Answers containing questions may cause misclassification
- Images must be processed in correct order

## Future Improvements
- EasyOCR integration for better handwriting recognition
- Multi-OCR voting system
- Layout-based segmentation
- Nested structure preservation
- GUI interface
