import os
from typing import List, Optional
from pathlib import Path
import mimetypes

# PDF processing
from PyPDF2 import PdfReader
import pypdf

# Image processing
from PIL import Image
import pytesseract

# Document processing
from docx import Document

# Excel and CSV processing
import pandas as pd
import openpyxl
import xlrd

# Utilities
import aiofiles


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
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")
        return text
    
    @staticmethod
    async def _extract_from_csv(file_path: str) -> str:
        """Extract text from CSV file"""
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
        """Extract text from Excel file with robust fallback mechanisms"""
        file_extension = Path(file_path).suffix.lower()
        file_name = Path(file_path).name
        
        # First, try to detect if it's actually a text-based file (CSV/HTML) disguised as Excel
        try:
            with open(file_path, 'rb') as f:
                first_bytes = f.read(1024)
                # Check if it's actually HTML or CSV
                if b'<html' in first_bytes.lower() or b'<table' in first_bytes.lower():
                    # It's HTML disguised as Excel
                    try:
                        dfs = pd.read_html(file_path)
                        if dfs:
                            text = f"Excel/HTML File: {file_name}\n\n"
                            for idx, df in enumerate(dfs):
                                text += f"\n{'='*50}\n"
                                text += f"Table {idx + 1}\n"
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
                    except:
                        pass
                
                # Check if it might be a CSV
                try:
                    first_text = first_bytes.decode('utf-8', errors='ignore')
                    if ',' in first_text or '\t' in first_text:
                        # Try reading as CSV
                        df = pd.read_csv(file_path)
                        text = f"Excel/CSV File: {file_name}\n\n"
                        text += f"Columns: {', '.join(df.columns.astype(str).tolist())}\n"
                        text += f"Number of rows: {len(df)}\n\n"
                        text += "Data:\n"
                        if len(df) > 1000:
                            text += df.head(1000).to_string(index=False)
                            text += f"\n\n... (showing first 1000 of {len(df)} rows)"
                        else:
                            text += df.to_string(index=False)
                        return text
                except:
                    pass
        except:
            pass
        
        # Now try standard Excel engines with proper file handle management
        text = f"Excel File: {file_name}\n\n"
        approaches = [
            ('openpyxl', 'openpyxl'),
            ('xlrd', 'xlrd'),
            ('pyxlsb', 'pyxlsb'),
            ('auto', None),
        ]
        
        last_error = None
        for approach_name, engine in approaches:
            try:
                if engine == 'pyxlsb' and file_extension != '.xlsb':
                    continue
                
                # Use context manager to ensure file is closed
                with pd.ExcelFile(file_path, engine=engine) as excel_file:
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
                last_error = e
                continue
        
        # Last resort: try to read as plain text
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
                if content.strip():
                    return f"Excel File (read as text): {file_name}\n\n{content[:50000]}"
        except:
            pass
        
        raise Exception(f"Could not read Excel file. The file may be corrupted, password-protected, or in an unsupported format. Last error: {str(last_error)}")
    
    @staticmethod
    async def _extract_from_image(file_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(file_path)
            # Use pytesseract for OCR
            text = pytesseract.image_to_string(image)
        except Exception as e:
            # If OCR fails, return a description
            text = f"[Image file: {Path(file_path).name}. OCR extraction failed: {str(e)}]"
        return text
    
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
