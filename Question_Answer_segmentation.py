import cv2
import pytesseract
import re
import numpy as np

# -------------------------------------------------------------------
# SET TESSERACT PATH (Windows only)
# -------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\Shruthi\Downloads\Tesseract-OCR\tesseract.exe"
)

# -------------------------------------------------------------------
# IMPROVED OCR PIPELINE FOR HANDWRITING
# -------------------------------------------------------------------
def source_pipeline(image_path):
    """
    Enhanced OCR pipeline optimized for handwritten text
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found:", image_path)
        return ""

    # Resize image for better OCR (if too small)
    height, width = img.shape[:2]
    if height < 1500:
        scale_factor = 1500 / height
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, 
                        interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive thresholding works better for handwriting
    adaptive_thresh = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    # Denoise
    denoised = cv2.fastNlMeansDenoising(adaptive_thresh, None, 10, 7, 21)

    # Light morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    # OCR - try PSM 6 (uniform block)
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(morph, config=config)
    
    return text


# -------------------------------------------------------------------
# ENHANCED QUESTION-ANSWER SEPARATION WITH PRIORITY RULES
# -------------------------------------------------------------------
def separate_qa_with_regex(image_list):
    """
    Separate questions and answers with strict priority rules:
    
    PRIORITY 1: Lines ending with '?' are ALWAYS questions
    PRIORITY 2: Lines starting with Q.) are questions (with subdivisions)
    PRIORITY 3: Empty lines signal transition from Q to A
    PRIORITY 4: Lines starting with A.) are answers (with subdivisions)
    """
    results = {"Questions": [], "Answers": []}
    current_mode = None
    current_text = ""
    previous_line_was_empty = False
    
    # Q.) Q) Q: Q. Q- markers (also handles OCR errors like B.)
    q_markers = r'^[QqBb][\.\)\:\-]\s*'
    
    # A.) A) A: A. A- Ans.) Answer: markers
    a_markers = r'^[Aa][\.\)\:\-]\s*|^[Aa]ns[\.\)\:\-]\s*|^[Aa]nswer[\.\)\:\-]\s*'
    
    # Subdivision markers: a) b) c) or a.) b.) c.) or (a) (b) (c)
    subdivision = r'^\s*[a-z]\s*[\.\)\:]\s*|^\s*\([a-z]\)\s*|^\s*[uw]\s*\)'
    
    def save_current_item():
        """Helper to save current item to results"""
        nonlocal current_text, current_mode
        if current_text and current_mode:
            current_text = re.sub(r'\s+', ' ', current_text).strip()
            if current_text:
                results[current_mode].append(current_text)
        current_text = ""
    
    for img_path in image_list:
        text = source_pipeline(img_path)
        lines = text.split('\n')
        
        for line in lines:
            original_line = line
            line = line.strip()
            
            # Check if this is an empty line (transition point)
            if not line:
                previous_line_was_empty = True
                continue
            
            # Very short lines - skip
            if len(line) < 2:
                continue
            
            # Clean but preserve important characters
            line = re.sub(r'[^\w\s\?\.\,\:\;\(\)\-]', '', line)
            line = line.strip()
            
            if not line:
                previous_line_was_empty = True
                continue
            
            # ================================================================
            # PRIORITY 1: LINE ENDS WITH '?' - ALWAYS A QUESTION
            # ================================================================
            if line.endswith('?'):
                # If we're currently building an answer, save it first
                if current_mode == "Answers" and current_text:
                    save_current_item()
                
                # If we're building a question, append to it
                if current_mode == "Questions" and current_text and not previous_line_was_empty:
                    current_text += " " + line
                else:
                    # Start a new question
                    save_current_item()
                    current_mode = "Questions"
                    current_text = line
                
                previous_line_was_empty = False
                continue
            
            # ================================================================
            # PRIORITY 2: LINE STARTS WITH Q.) - START OF QUESTION
            # ================================================================
            if re.match(q_markers, line, re.IGNORECASE):
                # Save whatever we were building
                save_current_item()
                
                # Start new question
                current_mode = "Questions"
                current_text = re.sub(q_markers, '', line, flags=re.IGNORECASE).strip()
                previous_line_was_empty = False
                continue
            
            # ================================================================
            # PRIORITY 4: LINE STARTS WITH A.) - START OF ANSWER
            # ================================================================
            if re.match(a_markers, line, re.IGNORECASE):
                # Save previous content
                save_current_item()
                
                # Start new answer
                current_mode = "Answers"
                current_text = re.sub(a_markers, '', line, flags=re.IGNORECASE).strip()
                previous_line_was_empty = False
                continue
            
            # ================================================================
            # CHECK FOR SUBDIVISION (a), b), c), etc.)
            # ================================================================
            if re.match(subdivision, line, re.IGNORECASE):
                # Subdivisions belong to the current item
                if current_text:
                    current_text += " " + line
                else:
                    current_text = line
                previous_line_was_empty = False
                continue
            
            # ================================================================
            # PRIORITY 3: EMPTY LINE TRANSITION - Switch from Q to A
            # ================================================================
            if previous_line_was_empty and current_mode == "Questions" and current_text:
                # We had a question, then empty line, now new content
                # This new content is likely an answer
                save_current_item()
                current_mode = "Answers"
                current_text = line
                previous_line_was_empty = False
                continue
            
            # ================================================================
            # REGULAR CONTINUATION LINE
            # ================================================================
            if current_mode:
                # Continue building current item
                if current_text:
                    current_text += " " + line
                else:
                    current_text = line
            else:
                # No mode set - default to question
                current_mode = "Questions"
                current_text = line
            
            previous_line_was_empty = False
        
        # Save the last item after processing all lines
        save_current_item()
    
    return results


# -------------------------------------------------------------------
# CLEAN OUTPUT FORMATTING
# -------------------------------------------------------------------
def print_results(output):
    """Print clean Q&A results"""
    print("\n" + "="*70)
    print("QUESTIONS")
    print("="*70)
    if not output["Questions"]:
        print("No questions detected")
    else:
        for i, q in enumerate(output["Questions"], 1):
            print(f"{i}. {q}")
    
    print("\n" + "="*70)
    print("ANSWERS")
    print("="*70)
    if not output["Answers"]:
        print("No answers detected")
    else:
        for i, a in enumerate(output["Answers"], 1):
            print(f"{i}. {a}")
    
    print("\n" + "="*70)


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Your image path
    my_images = [
        r"C:\Users\Shruthi\OneDrive\Desktop\Handwriting_OCR\Image.jpeg"
    ]

    # Process images and display results
    output = separate_qa_with_regex(my_images)
    print_results(output)
