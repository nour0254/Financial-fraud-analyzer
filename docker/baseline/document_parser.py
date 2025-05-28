# Phase 2: Document Parsing with NLP
# Quick Implementation: OCR + Regex Approach
# document_parser.py
import cv2
import easyocr
import re
import pandas as pd
from datetime import datetime
import numpy as np

class DocumentParser:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def extract_text_from_image(self, image_path):
        """Extract text using EasyOCR"""
        results = self.reader.readtext(image_path)
        return ' '.join([result[1] for result in results])

    def extract_fields_from_text(self, text):
        """Extract key fields using regex patterns"""

        # Amount patterns
        amount_patterns = [
            r'\$[\d,]+\.?\d*',
            r'Total[:\s]*\$?[\d,]+\.?\d*',
            r'Amount[:\s]*\$?[\d,]+\.?\d*'
        ]

        # Date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}'
        ]

        # Invoice ID patterns
        invoice_patterns = [
            r'Invoice[#\s]*:?\s*([A-Z0-9-]+)',
            r'INV[#\s]*:?\s*([A-Z0-9-]+)',
            r'#([A-Z0-9-]+)'
        ]

        extracted = {
            'amounts': [],
            'dates': [],
            'invoice_ids': [],
            'vendors': []
        }

        # Extract amounts
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted['amounts'].extend(matches)

        # Extract dates
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted['dates'].extend(matches)

        # Extract invoice IDs
        for pattern in invoice_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted['invoice_ids'].extend(matches)

        return extracted

    def parse_document(self, file_path):
        """Main parsing function"""
        try:
            # Extract text
            text = self.extract_text_from_image(file_path)

            # Extract fields
            fields = self.extract_fields_from_text(text)

            # Clean and structure data
            result = {
                'raw_text': text,
                'invoice_id': fields['invoice_ids'][0] if fields['invoice_ids'] else 'Unknown',
                'amount': self._clean_amount(fields['amounts'][0]) if fields['amounts'] else 0,
                'date': self._clean_date(fields['dates'][0]) if fields['dates'] else None,
                'extraction_confidence': self._calculate_confidence(fields)
            }

            return result

        except Exception as e:
            return {'error': str(e)}

    def _clean_amount(self, amount_str):
        """Clean amount string to float"""
        cleaned = re.sub(r'[^\d.]', '', amount_str)
        try:
            return float(cleaned)
        except:
            return 0.0

    def _clean_date(self, date_str):
        """Clean date string"""
        # Simple date cleaning - expand as needed
        return date_str.strip()

    def _calculate_confidence(self, fields):
        """Calculate extraction confidence score"""
        score = 0
        if fields['amounts']: score += 0.4
        if fields['dates']: score += 0.3
        if fields['invoice_ids']: score += 0.3
        return score

# Usage example
parser = DocumentParser()
result = parser.parse_document('sample_invoice.jpg')
print(result)
