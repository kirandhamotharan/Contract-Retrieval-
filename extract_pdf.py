import PyPDF2
import sys

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- PAGE {page_num + 1} ---\n"
                text += page_text
            
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_pdf.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    text = extract_text_from_pdf(pdf_path)
    
    if text:
        print(text)
        # Save to file
        output_file = "extracted_text.txt"
        with open(output_file, 'w') as f:
            f.write(text)
        print(f"\nText saved to {output_file}")