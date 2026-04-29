import pytesseract
import pandas as pd
import re
import os
import platform
import time
from pdf2image import convert_from_path
from pdfminer.pdfpage import PDFPage
from io import StringIO
import shutil

# --- CONFIGURE TESSERACT PATH FOR WINDOWS ---
# Pointing specifically to the executable on your D: drive
pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'D:\Tesseract-OCR\tessdata'

# --- CONFIGURATION & UTILITY FUNCTIONS ---

def get_pdf_page_count(file_path):
    """Safely gets total page count without loading the whole PDF into RAM."""
    try:
        with open(file_path, 'rb') as in_file:
            count = len(list(PDFPage.get_pages(in_file)))
        return count
    except Exception as e:
        print(f"Warning: Could not get page count. Error: {e}")
        return 0

def clean_text(text):
    """Post-processing for Albanian OCR artifacts."""
    if not text:
        return ""

    # Fix broken newlines that split words (e.g., "shqip- tar" -> "shqiptar")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Remove excessive blank lines
    text = re.sub(r'\n\s*\n', '\n', text)

    # --- ADVANCED ALBANIAN CHARACTER FIXES ---
    replacements = {
        '3': 'ë', '¢': 'ç', '£': 'Ë', '€': 'ë', 
        '©': 'ç', '®': 'ë', 'ﬁ': 'fi', 'ﬂ': 'fl'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.strip()

def process_dictionary(folder_path):
    data = []
    # Only process PDF files
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    if not files:
        print(f"Error: No PDF files found in {folder_path}.")
        return pd.DataFrame(data)

    # --- Poppler Path Verification ---
    poppler_bin_path = r'D:\poppler-25.12.0\Library\bin'
    if not os.path.exists(poppler_bin_path):
        # Fallback to checking system PATH
        if not shutil.which('pdftoppm'):
            print(f"ERROR: Poppler not found at {poppler_bin_path} or in PATH.")
            return pd.DataFrame(data)
        poppler_bin_path = None 

    print(f"✅ Environment ready. Processing {len(files)} files...")

    for filename in files:
        start_time = time.time()
        print(f"\nProcessing: {filename}")
        file_path = os.path.join(folder_path, filename)
        
        # PSM 6 is often best for uniform dictionary blocks
        config_custom = r'--oem 3 --psm 6 -l sqi'
        page_count = get_pdf_page_count(file_path)
        
        if page_count == 0: continue

        full_text = ""
        for page_num in range(1, page_count + 1):
            print(f"  -> Progress: {page_num}/{page_count} pages...", end="\r")

            try:
                # Convert only ONE page at a time to save RAM
                kwargs = {'dpi': 300, 'first_page': page_num, 'last_page': page_num}
                if poppler_bin_path:
                    kwargs['poppler_path'] = poppler_bin_path
                
                page_images = convert_from_path(file_path, **kwargs)
                if not page_images: continue
                
                # Perform OCR
                raw_text = pytesseract.image_to_string(page_images[0], config=config_custom)
                full_text += raw_text + "\n"
                
                # Explicit memory cleanup
                del page_images

            except Exception as e:
                print(f"\n  -> Error on Page {page_num}: {e}")
                break

        # Post-processing and saving data
        if full_text:
            cleaned = clean_text(full_text)
            data.append({
                'filename': filename,
                'raw_text': full_text,
                'cleaned_text': cleaned,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
        duration = time.time() - start_time
        print(f"\n  Done! Took {duration:.2f} seconds.")

    return pd.DataFrame(data)

# --- EXECUTION ---
if __name__ == "__main__":
    # Ensure this matches your folder name on the D: drive
    target_folder = "./scanned pages/" 
    
    if not os.path.exists(target_folder):
        print(f"Creating folder: {target_folder}. Please place your PDFs there.")
        os.makedirs(target_folder)
    else:
        results_df = process_dictionary(target_folder)

        if not results_df.empty:
            output_file = 'albanian_dictionary_dataset.csv'
            results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n🚀 SUCCESS! Dataset saved to: {output_file}")
        else:
            print("\nProcessing failed. Check if PDFs are valid and paths are correct.")