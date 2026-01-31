import os
from typing import List, Optional
from pathlib import Path
import mimetypes

# Core utilities (always available)
import aiofiles

# Heavy dependencies - import only when needed to avoid serverless startup failures
# These will be imported lazily in the methods that use them


class FileProcessor:
    """Process various file types and extract text content"""
    
    @staticmethod
    async def extract_text_from_file(file_path: str) -> str:
        """
        Extract text from various file types
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.txt':
                return await FileProcessor._extract_from_txt(file_path)
            elif file_extension == '.pdf':
                return await FileProcessor._extract_from_pdf(file_path)
            elif file_extension in ['.doc', '.docx']:
                return await FileProcessor._extract_from_docx(file_path)
            elif file_extension == '.csv':
                return await FileProcessor._extract_from_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                return await FileProcessor._extract_from_excel(file_path)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                return await FileProcessor._extract_from_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            raise Exception(f"Error processing file {file_path}: {str(e)}")
    
    @staticmethod
    async def _extract_from_txt(file_path: str) -> str:
        """Extract text from text file"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = await f.read()
        return content
    
    @staticmethod
    async def _extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise Exception("PyPDF2 not installed. Install with: pip install PyPDF2")
        
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
        return text
    
    @staticmethod
    async def _extract_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
        except ImportError:
            raise Exception("python-docx not installed. Install with: pip install python-docx")
        
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")
        return text
    
    @staticmethod
    async def _extract_from_csv(file_path: str) -> str:
        """Extract text from CSV file"""
        try:
            import pandas as pd
        except ImportError:
            # Fallback to basic CSV reading if pandas not available
            import csv
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()
                return f"CSV File: {Path(file_path).name}\n\n{content}"
            except Exception as e:
                raise Exception(f"Error reading CSV: {str(e)}")
        
        try:
            df = pd.read_csv(file_path)
            # Convert DataFrame to a readable text format
            text = f"CSV File: {Path(file_path).name}\n\n"
            text += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            text += f"Number of rows: {len(df)}\n\n"
            text += "Data:\n"
            text += df.to_string(index=False)
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from CSV: {str(e)}")
    
    @staticmethod
    async def _extract_from_excel(file_path: str) -> str:
        """Extract text from Excel file - requires pandas (optional dependency)"""
        try:
            import pandas as pd
        except ImportError:
            raise Exception("Excel support requires pandas. Install with: pip install pandas openpyxl")
        
        file_name = Path(file_path).name
        text = f"Excel File: {file_name}\n\n"
        
        try:
            # Try reading Excel file
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                text += f"\n{'='*50}\n"
                text += f"Sheet: {sheet_name}\n"
                text += f"Columns: {', '.join(df.columns.astype(str).tolist())}\n"
                text += f"Number of rows: {len(df)}\n\n"
                text += "Data:\n"
                if len(df) > 1000:
                    text += df.head(1000).to_string(index=False)
                    text += f"\n\n... (showing first 1000 of {len(df)} rows)"
                else:
                    text += df.to_string(index=False)
                text += f"\n{'='*50}\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from Excel: {str(e)}")
    
    @staticmethod
    async def _extract_from_image(file_path: str) -> str:
        """Extract text from image - OCR not supported in serverless (optional)"""
        file_name = Path(file_path).name
        try:
            from PIL import Image
        except ImportError:
            return f"[Image file: {file_name}. Image processing requires Pillow library]"
        
        # Note: pytesseract requires system binaries and is not suitable for serverless
        # Return basic info instead
        try:
            image = Image.open(file_path)
            return f"[Image file: {file_name}, Size: {image.size[0]}x{image.size[1]}, Mode: {image.mode}. OCR not available in serverless environment]"
        except Exception as e:
            return f"[Image file: {file_name}. Could not process: {str(e)}]"
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
        """Validate if file extension is allowed"""
        extension = Path(filename).suffix.lower().lstrip('.')
        return extension in allowed_extensions
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """Get file size in MB"""
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
