from __future__ import annotations

import asyncio
import functools
import re
import json
import threading
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
from datetime import datetime
from difflib import SequenceMatcher
from PIL import Image
from pydantic import BaseModel, Field, validator
from enum import Enum
import os
import tempfile
import shutil

# Optional imports for OCR - handle gracefully if not available
PADDLEOCR_IMPORT_ERROR: Optional[Exception] = None
PaddleOCR = None
try:
    from paddleocr import PaddleOCR
except ImportError as exc:
    PADDLEOCR_IMPORT_ERROR = exc

# Optional imports for Pathway parsers
PATHWAY_PARSER_IMPORT_ERROR: Optional[Exception] = None
PaddleOCRParser = None
try:
    from pathway.xpacks.llm.parsers import PaddleOCRParser
    import pathway as pw
except ImportError as exc:
    PATHWAY_PARSER_IMPORT_ERROR = exc
    pw = None

# Optional imports for FastAPI
FASTAPI_IMPORT_ERROR: Optional[Exception] = None
try:
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import JSONResponse
except ImportError as exc:
    FASTAPI_IMPORT_ERROR = exc
    FastAPI = None
    File = None
    UploadFile = None
    Form = None
    HTTPException = None
    JSONResponse = None

# Optional import for pdf2image
PDF2IMAGE_IMPORT_ERROR: Optional[Exception] = None
try:
    from pdf2image import convert_from_bytes
except ImportError as exc:
    PDF2IMAGE_IMPORT_ERROR = exc
    convert_from_bytes = None

VIDEO_MODULE_IMPORT_ERROR: Optional[Exception] = None
try:  # pragma: no cover - heavy deps checked at runtime
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from torchvision import transforms  # type: ignore
    from facenet_pytorch import MTCNN, InceptionResnetV1  # type: ignore
except ImportError as exc:  # pragma: no cover - handled gracefully later
    VIDEO_MODULE_IMPORT_ERROR = exc
    cv2 = None  # type: ignore
    np = None  # type: ignore
    torch = None  # type: ignore
    transforms = None  # type: ignore
    MTCNN = None  # type: ignore
    InceptionResnetV1 = None  # type: ignore

# Optional import for kyc_master_store
try:
    from kyc_master_store import register_master_json
except ImportError:
    register_master_json = None

 # Logging removed to avoid PII exposure

# Initialize FastAPI app (only if FastAPI is available)
if FastAPI is not None:
    app = FastAPI(title="Integrated KYC Verification API", version="2.0.0")
else:
    app = None

# ============================================================================
# ENUMS AND MODELS
# ============================================================================

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    TRANSGENDER = "Transgender"
    OTHER = "Other"

class MaritalStatus(str, Enum):
    SINGLE = "Single"
    MARRIED = "Married"
    DIVORCED = "Divorced"
    WIDOWED = "Widowed"

class QuestionAnswer(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"

class PANCardInput(BaseModel):
    pan_number: Optional[str] = None
    name: Optional[str] = None
    father_name: Optional[str] = None
    dob: Optional[str] = None
    raw_text: Optional[str] = None

class AadhaarCardInput(BaseModel):
    aadhaar_number: Optional[str] = None
    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[Gender] = None

class ITRDocumentInput(BaseModel):
    PAN: Optional[str] = None
    Name: Optional[str] = None
    Status: Optional[str] = None
    ITR_Type: Optional[str] = None
    Filed_u_s: Optional[str] = None
    Total_Income: Optional[str] = None
    Taxes_Paid: Optional[str] = None

class QuestionnaireInput(BaseModel):
    Q1: Optional[QuestionAnswer] = None
    Q2: Optional[QuestionAnswer] = None
    Q3: Optional[QuestionAnswer] = None
    Q4: Optional[QuestionAnswer] = None
    Q5: Optional[QuestionAnswer] = None
    Q6: Optional[QuestionAnswer] = None

class AdditionalDetails(BaseModel):
    amount_to_invest: Optional[float] = None
    address: Optional[str] = None
    main_occupation: Optional[str] = None
    marital_status: Optional[MaritalStatus] = None
    dependents: Optional[int] = None
    citizenship: Optional[str] = None

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

MAX_IMAGE_SIDE = 3000  # Maximum side length to avoid PaddleOCR resizing warnings

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Preprocess image before OCR to avoid resizing warnings.
    Resizes image if any side exceeds MAX_IMAGE_SIDE while maintaining aspect ratio.
    """
    width, height = img.size
    max_side = max(width, height)
    
    if max_side > MAX_IMAGE_SIDE:
        # Calculate scale factor
        scale = MAX_IMAGE_SIDE / max_side
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return img

# ============================================================================
# PARSER FUNCTIONS
# ============================================================================

def parser_paddleocr(img):
    """PaddleOCR Parser - extracts text from image (synchronous version)"""
    if PaddleOCR is None:
        raise ImportError(
            "PaddleOCR is not installed. Please install it with: pip install paddleocr paddlepaddle"
        )
    
    # Preprocess image to avoid resizing warnings
    img = preprocess_image_for_ocr(img)
    
    paddle_ocr = PaddleOCR(
        lang='en',
        text_det_box_thresh=0.3,
        text_det_unclip_ratio=2.0,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

    # Try to use PaddleOCRParser from pathway if available (like combined_api.py)
    if PaddleOCRParser is not None:
        try:
            # Initialize PaddleOCRParser with PaddleOCR pipeline
            ocr_parser = PaddleOCRParser(pipeline=paddle_ocr, concatenate_pages=False)
            
            buf = BytesIO()
            img.save(buf, format='PNG')
            img_bytes = buf.getvalue()

            # Run sync - PaddleOCRParser has __wrapped__ method for sync usage
            # Run async in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(ocr_parser.__wrapped__(img_bytes))
            finally:
                loop.close()
            
            # Extract text from all chunks
            if isinstance(results, list):
                full_text = "\n".join([chunk[0] if isinstance(chunk, tuple) else str(chunk) for chunk in results])
                return full_text
            
            return str(results)
        except Exception:
            # Fall back to direct PaddleOCR if PaddleOCRParser fails
            pass
    
    # Fallback: Use PaddleOCR directly
    # Convert PIL Image to bytes and save to temp file
    buf = BytesIO()
    img.save(buf, format='PNG')
    img_bytes = buf.getvalue()

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_file.write(img_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Run OCR directly
        result = paddle_ocr.predict(tmp_path)

        # Extract text from OCR result. Newer PaddleOCR (>=3.x) returns a list of
        # OCRResult dict-like objects with `rec_texts`; older versions return the
        # nested list structure. Handle both to stay compatible.
        text_lines: List[str] = []
        if isinstance(result, list):
            # Newer structure: list of OCRResult objects/dicts
            if result and isinstance(result[0], dict):
                for page in result:
                    rec_texts = page.get("rec_texts") or page.get("text")
                    if isinstance(rec_texts, list):
                        text_lines.extend(str(t) for t in rec_texts if t)
                    elif rec_texts:
                        text_lines.append(str(rec_texts))
            # Legacy structure: list with a single list of (bbox, (text, score))
            elif result and isinstance(result[0], (list, tuple)):
                for line in result[0]:
                    if len(line) >= 2 and line[1]:
                        text = line[1][0] if isinstance(line[1], tuple) else line[1]
                        text_lines.append(str(text))

        return "\n".join(text_lines)
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def parser_docling(img):
    """Docling Hybrid Parser - extracts text from image using Docling + Unstructured (synchronous version)"""
    
    # Preprocess image to avoid resizing warnings
    img = preprocess_image_for_ocr(img)
    
    # ====================================================================
    # UNIVERSAL HYBRID PARSER (Docling + Unstructured)
    # ====================================================================

    class HybridDocumentParser:
        def __init__(
            self,
            docling_parser=None,
            similarity_threshold: float = 0.6,
            min_word_length: int = 8,
            max_window_size: int = 5,
            aggressive_split: bool = True,
        ):
            if docling_parser is None:
                try:
                    from pathway.xpacks.llm.parsers import DoclingParser
                    docling_parser = DoclingParser(
                        image_parsing_strategy="docling",
                        table_parsing_strategy="docling",
                        multimodal_llm=None,
                    )
                except Exception:
                    docling_parser = None

            self.docling_parser = docling_parser
            self.similarity_threshold = similarity_threshold
            self.min_word_length = min_word_length
            self.max_window_size = max_window_size
            self.aggressive_split = aggressive_split

        def _normalize_text(self, text: str) -> str:
            return re.sub(r'[^a-z0-9]', '', text.lower())

        def _calculate_similarity(self, str1: str, str2: str) -> float:
            return SequenceMatcher(None, str1, str2).ratio()

        def _extract_with_unstructured(self, contents: bytes) -> str:
            try:
                import unstructured.partition.auto
                elements = unstructured.partition.auto.partition(file=BytesIO(contents))
                full_text = "\n".join([str(element) for element in elements])
                return full_text
            except ImportError:
                return ""

        def _build_word_index(self, text: str) -> Dict[str, List[str]]:
            words = text.split()
            index = {}

            for i in range(len(words)):
                for window_size in range(1, min(self.max_window_size + 1, len(words) - i + 1)):
                    phrase = ' '.join(words[i:i+window_size])
                    normalized = self._normalize_text(phrase)

                    if normalized not in index:
                        index[normalized] = []

                    if phrase not in index[normalized]:
                        index[normalized].append(phrase)

            return index

        def _find_best_match(self, docling_word: str, word_index: Dict[str, List[str]]) -> str:
            normalized_docling = self._normalize_text(docling_word)

            if len(normalized_docling) < self.min_word_length or ' ' in docling_word:
                return docling_word

            if normalized_docling in word_index:
                candidates = word_index[normalized_docling]
                for candidate in candidates:
                    if ' ' in candidate:
                        return candidate
                return candidates[0]

            best_match = None
            best_similarity = 0

            for normalized_key, candidates in word_index.items():
                if abs(len(normalized_key) - len(normalized_docling)) > len(normalized_docling) * 0.5:
                    continue

                similarity = self._calculate_similarity(normalized_docling, normalized_key)

                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    for candidate in candidates:
                        if ' ' in candidate:
                            best_match = candidate
                            break
                    if not best_match:
                        best_match = candidates[0]

            if not best_match and self.aggressive_split and len(normalized_docling) >= 15:
                best_match = self._smart_split(docling_word)

            return best_match if best_match else docling_word

        def _smart_split(self, word: str) -> str:
            if any(c.isupper() for c in word[1:]):
                parts = re.findall(r'[A-Z][a-z]*|[a-z]+', word)
                if len(parts) > 1:
                    return ' '.join(parts)
            return word

        def _clean_artifacts(self, text: str) -> str:
            text = re.sub(r'^having\s*\n', '', text, flags=re.MULTILINE)
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            return text.strip()

        def _apply_spacing_correction(self, docling_text: str, word_index: Dict[str, List[str]]) -> str:
            lines = docling_text.split('\n')
            corrected_lines = []

            for line in lines:
                if not line.strip():
                    corrected_lines.append(line)
                    continue

                if line.strip().startswith('#'):
                    match = re.match(r'^(#+\s*)', line)
                    if match:
                        header_marker = match.group(1)
                        content = line[len(header_marker):]
                        words = content.split()
                        corrected_words = [self._find_best_match(word, word_index) for word in words]
                        corrected_lines.append(header_marker + ' '.join(corrected_words))
                    else:
                        corrected_lines.append(line)
                    continue

                words = line.split()
                corrected_words = [self._find_best_match(word, word_index) for word in words]
                corrected_lines.append(' '.join(corrected_words))

            result = '\n'.join(corrected_lines)
            return self._clean_artifacts(result)

        def _parse_sync(self, contents: bytes) -> Dict[str, Any]:
            """Synchronous parsing method"""
            # Try to use docling parser if available
            if self.docling_parser is not None:
                try:
                    # Run async parser in event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        docling_result = loop.run_until_complete(self.docling_parser.parse(contents))
                    finally:
                        loop.close()
                except Exception:
                    docling_result = []
            else:
                docling_result = []
            
            unstructured_text = self._extract_with_unstructured(contents)
            word_index = self._build_word_index(unstructured_text)

            corrected_chunks = []
            for chunk, metadata in docling_result:
                corrected_chunk = self._apply_spacing_correction(chunk, word_index)
                corrected_chunks.append((corrected_chunk, metadata))

            if corrected_chunks:
                combined_text = "\n".join([chunk for chunk, _ in corrected_chunks])
            else:
                # Fallback to unstructured text if docling failed
                combined_text = unstructured_text

            return {
                'parsed_chunks': corrected_chunks,
                'full_text': combined_text,
                'docling_text': "\n".join([chunk for chunk, _ in docling_result]) if docling_result else "",
                'unstructured_text': unstructured_text
            }

    buf = BytesIO()
    img.save(buf, format='PNG')
    img_bytes = buf.getvalue()

    # Call the sync method directly
    parser = HybridDocumentParser()
    results = parser._parse_sync(img_bytes)

    return results['full_text']


# ============================================================================
# PAN CARD EXTRACTOR
# ============================================================================

class PANCardExtractor:
    """Extract PAN card details from OCR output using regex patterns"""

    def __init__(self):
        # PAN number pattern: 5 letters, 4 digits, 1 letter
        self.pan_patterns = [
            r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',
            r"[A-Z]{5}[0-9]{4}[A-Z](?=\s|$)"
        ]

        # Name pattern - Match uppercase words and spaces in between
        self.name_patterns = [
            r'\b[A-Z][A-Z\s]{2,50}\b',
            r"\s*([A-Z\s]+?)(?=\s*)",
        ]

        # Date patterns for DOB
        self.date_patterns = [
            r'\b(\d{2})[/-](\d{2})[/-](\d{4})\b',
            r'\b(\d{2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b',
            r"\b(\d{2})\s*([A-Za-z]+)\s*(\d{4})\b",
        ]

        # Father's name pattern
        self.father_patterns = [
            r"\s*([A-Z\s]+)(?=\n|\s|)",
            r"\s*[:\-]?\s*([A-Z\s]+)",
        ]

    def clean_text(self, text):
        """Clean and normalize OCR text"""
        text = ' '.join(text.split())
        text = text.replace('0', 'O').replace('1', 'I')
        return text

    def extract_pan_number(self, text):
        """Extract PAN number from text after 'Permanent Account Number'"""
        start_marker = "Permanent Account Number Card"
        start_pos = text.find(start_marker)
        if start_pos != -1:
            pan_text = text[start_pos + len(start_marker):].strip()
            best_match = None
            for pattern in self.pan_patterns:
                matches = re.findall(pattern, pan_text)
                if matches:
                    best_match = matches[0]
                    break
            return best_match
        return None

    def extract_name(self, text):
        """Extract Name from text between 'Name' and 'Father's Name'"""
        start_marker = "Name"
        end_marker = "Father's Name"
        start_pos = text.find(start_marker)
        end_pos = text.find(end_marker)

        if start_pos != -1 and end_pos != -1:
            name_text = text[start_pos + len(start_marker):end_pos].strip()
            best_match = None
            for pattern in self.name_patterns:
                matches = re.findall(pattern, name_text)
                if matches:
                    best_match = matches[0].strip()
                    break
            return best_match
        return None

    def extract_father_name(self, text, name=None):
        """Extract Father's Name from text"""
        start_marker = "Father's Name"
        end_marker = "Date of Birth"
        start_pos = text.find(start_marker)
        end_pos = text.find(end_marker)

        if start_pos != -1 and end_pos != -1:
            father_name_text = text[start_pos + len(start_marker):end_pos].strip()
            best_match = None
            for pattern in self.father_patterns:
                match = re.search(pattern, father_name_text, re.IGNORECASE)
                if match:
                    father_name = match.group(1).strip()
                    if name and father_name != name:
                        best_match = father_name
                        break
            return best_match
        return None

    def extract_dob(self, text):
        """Extract Date of Birth from text"""
        start_marker = "Date of Birth"
        start_pos = text.find(start_marker)
        if start_pos != -1:
            dob_text = text[start_pos + len(start_marker):].strip()
            best_match = None
            for pattern in self.date_patterns:
                matches = re.findall(pattern, dob_text, re.IGNORECASE)
                if matches:
                    best_match = matches[0] if isinstance(matches[0], str) else '/'.join(matches[0])
                    break
            return best_match
        return None

    def extract_all(self, ocr_output):
        """Extract all information from OCR output"""
        if isinstance(ocr_output, list):
            text = self.paddleocr_to_text(ocr_output)
        else:
            text = ocr_output

        name = self.extract_name(text)
        result = {
            'pan_number': self.extract_pan_number(text),
            'name': name,
            'father_name': self.extract_father_name(text, name),
            'dob': self.extract_dob(text),
            'raw_text': text
        }

        return result

    def paddleocr_to_text(self, ocr_result):
        """Convert PaddleOCR output format to plain text"""
        text_lines = []
        for item in ocr_result:
            if len(item) >= 2:
                text = item[1][0] if isinstance(item[1], tuple) else item[1]
                text_lines.append(text)
        return '\n'.join(text_lines)


# ============================================================================
# AADHAAR EXTRACTOR
# ============================================================================

class AadhaarExtractor:
    """Extract Aadhaar card information using regex"""

    BLACKLIST_WORDS = {
        'government', 'india', 'aadhaar', 'aadhar', 'adhaar', 'adhar',
        'female', 'male', 'dob', 'birth', 'date', 'card', 'number',
        'uid', 'unique', 'identification', 'authority', 'enrollment',
        'resident', 'indian', 'भारत', 'सरकार', 'आधार', 'update', 'help',
        'uidai', 'address', 'year', 'sex', 'age', 'to', 'of', 'the',
        'and', 'or', 'in', 'at', 'for', 'by', 'with', 'from'
    }

    def __init__(self):
        self.aadhaar_patterns = [
            r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b',
            r'\b\d{12}\b',
            r'\b\d{4}[\/\.\-\s]\d{4}[\/\.\-\s]\d{4}\b',
        ]

        self.date_patterns = [
            r'\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})\b',
            r'\b(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})\b',
            r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b',
        ]

        self.gender_pattern = r'\b(male|female|transgender|other)\b'

    def _is_blacklisted(self, word: str) -> bool:
        return word.lower().strip() in self.BLACKLIST_WORDS

    def _is_valid_name_word(self, word: str) -> bool:
        if len(word) < 2:
            return False
        if not any(c.isalpha() for c in word):
            return False
        if sum(c.isdigit() for c in word) > len(word) / 2:
            return False
        return True

    def extract_aadhaar_number(self, text: str) -> Optional[str]:
        for pattern in self.aadhaar_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                cleaned = re.sub(r'[^\d]', '', match)
                if len(cleaned) == 12 and len(set(cleaned)) > 1:
                    formatted = f"{cleaned[0:4]} {cleaned[4:8]} {cleaned[8:12]}"
                    return formatted
        return None

    def extract_date_of_birth(self, text: str) -> Optional[str]:
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if re.search(r'\b(dob|birth|date\s+of\s+birth|yob|year\s+of\s+birth)\b', line, re.IGNORECASE):
                search_text = '\n'.join(lines[i:min(i+3, len(lines))])
                for pattern in self.date_patterns:
                    matches = re.findall(pattern, search_text, re.IGNORECASE)
                    for match in matches:
                        try:
                            if len(match) == 3:
                                if match[1].isalpha():
                                    month_map = {
                                        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                                        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                                        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                                    }
                                    month_str = match[1][:3].lower()
                                    if month_str in month_map:
                                        day, month, year = int(match[0]), month_map[month_str], int(match[2])
                                        date_obj = datetime(year, month, day)
                                        formatted = date_obj.strftime("%d/%m/%Y")
                                        return formatted
                                elif len(match[0]) == 4:
                                    year, month, day = int(match[0]), int(match[1]), int(match[2])
                                    date_obj = datetime(year, month, day)
                                    formatted = date_obj.strftime("%d/%m/%Y")
                                    return formatted
                                else:
                                    day, month, year = int(match[0]), int(match[1]), int(match[2])
                                    date_obj = datetime(year, month, day)
                                    formatted = date_obj.strftime("%d/%m/%Y")
                                    return formatted
                        except (ValueError, IndexError):
                            continue
        return None

    def extract_gender(self, text: str) -> Optional[str]:
        match = re.search(self.gender_pattern, text, re.IGNORECASE)
        if match:
            gender = match.group(1).capitalize()
            return gender
        return None

    def extract_name(self, text: str) -> Optional[str]:
        lines = text.split('\n')
        candidate_names = []

        for line in lines:
            if re.search(r'(government|aadhaar|authority|unique|identification)', line, re.IGNORECASE):
                continue

            words = line.split()
            current_name = []

            for word in words:
                cleaned_word = re.sub(r'[^\w\s]', '', word).strip()

                if not cleaned_word or not self._is_valid_name_word(cleaned_word):
                    if current_name:
                        candidate_names.append(' '.join(current_name))
                        current_name = []
                    continue

                if self._is_blacklisted(cleaned_word):
                    if current_name:
                        candidate_names.append(' '.join(current_name))
                        current_name = []
                    continue

                if cleaned_word.isupper() or (cleaned_word[0].isupper() and cleaned_word[1:].islower()):
                    current_name.append(cleaned_word)
                else:
                    if current_name:
                        candidate_names.append(' '.join(current_name))
                        current_name = []

            if current_name:
                candidate_names.append(' '.join(current_name))

        valid_names = []
        for name in candidate_names:
            words = name.split()
            if len(words) >= 2:
                if all(self._is_valid_name_word(w) and not self._is_blacklisted(w) for w in words):
                    valid_names.append(name)

        if valid_names:
            best_name = max(valid_names, key=len)
            return best_name
        return None

    def extract_all(self, hybrid_text: str) -> Dict[str, Optional[str]]:
        results = {
            'aadhaar_number': self.extract_aadhaar_number(hybrid_text),
            'name': self.extract_name(hybrid_text),
            'date_of_birth': self.extract_date_of_birth(hybrid_text),
            'gender': self.extract_gender(hybrid_text),
        }
        return results


# ============================================================================
# ITR EXTRACTOR
# ============================================================================

def extract_between(text, start_kw, end_kw):
    """Extracts the substring between two keywords (case-insensitive)"""
    s = re.search(re.escape(start_kw), text, flags=re.I)
    e = re.search(re.escape(end_kw), text, flags=re.I)
    if not s:
        return ""
    start = s.end()
    end = e.start() if e else len(text)
    return text[start:end].strip()


def extract_itr_details(text):
    """Extracts specific details from an ITR document text"""
    # i) PAN
    pan_segment = extract_between(text, "PAN", "Name")
    pan_candidates = re.findall(r"[A-Z0-9]+", pan_segment)
    pan = max(pan_candidates, key=len) if pan_candidates else None

    # ii) Name
    name_segment = extract_between(text, "Name", "Address")
    name_matches = re.findall(r"[A-Za-z ]+", name_segment)
    name_matches = [m.strip() for m in name_matches if m.strip()]
    name = max(name_matches, key=len) if name_matches else None

    # iii) Total Income
    total_income_segment = extract_between(text, "Total Income", "Book Profit")
    num_candidates_total = re.findall(r"[\d,]+", total_income_segment)
    total_income = max(num_candidates_total, key=len).replace(",", "") if num_candidates_total else None

    # iv) Taxes Paid
    taxes_paid_segment = extract_between(text, "Taxes Paid", "(+)Tax Payable")
    num_candidates_tax = re.findall(r"[\d,]+", taxes_paid_segment)
    taxes_paid = max(num_candidates_tax, key=len).replace(",", "") if num_candidates_tax else None

    # v) ITR Type
    itr_segment = extract_between(text, "Form Number", "Filed u/s")
    itr_matches = re.findall(r"ITR-?\s*\d+", itr_segment, flags=re.I)
    itr_type = itr_matches[0].replace(" ", "").upper() if itr_matches else None

    # vi) Status
    status_segment = extract_between(text, "Status", "Form Number")
    status_match = re.findall(r"[A-Za-z]+", status_segment)
    status = status_match[0] if status_match else None

    # vii) Filed u/s
    filed_segment = extract_between(text, "Filed u/s", "e-Filing Acknowledgement")
    filed_us = filed_segment.strip() if filed_segment else None

    result = {
        "PAN": pan,
        "Name": name,
        "Status": status,
        "ITR_Type": itr_type,
        "Filed_u_s": filed_us,
        "Total_Income": total_income,
        "Taxes_Paid": taxes_paid
    }

    return result


# ============================================================================
# EXTRACTION WRAPPER FUNCTIONS
# ============================================================================

def pan_extract(raw_text: str) -> Dict[str, Any]:
    """Extract PAN card information from raw text"""
    extractor = PANCardExtractor()
    return extractor.extract_all(raw_text)


def adhaar_extract(raw_text: str) -> Dict[str, Any]:
    """Extract Aadhaar card information from raw text"""
    extractor = AadhaarExtractor()
    return extractor.extract_all(raw_text)


def itr_extract(raw_text: str) -> Dict[str, Any]:
    """Extract ITR information from raw text"""
    return extract_itr_details(raw_text)


# ============================================================================
# GLOBAL PARSER FUNCTION
# ============================================================================

def parser_(image: Image.Image, doc_type: str) -> Dict[str, Any]:
    """
    Global parser function that routes to appropriate parser and extractor (synchronous version)
    
    Args:
        image: PIL Image object
        doc_type: Type of document - 'pan', 'adhaar', or 'itr'
    
    Returns:
        Dictionary containing extracted information
    """
    doc_type = doc_type.lower()
    
    if doc_type == 'pan':
        raw_text = parser_paddleocr(image)
        result = pan_extract(raw_text)
        return result
    
    elif doc_type == 'adhaar':
        raw_text = parser_docling(image)
        result = adhaar_extract(raw_text)
        return result
    
    elif doc_type == 'itr':
        raw_text = parser_paddleocr(image)
        result = itr_extract(raw_text)
        return result
    
    else:
        raise ValueError(f"Invalid document type: {doc_type}. Must be one of: 'pan', 'adhaar', 'itr'")


# ============================================================================
# HELPER FUNCTIONS FOR CROSS-VERIFICATION
# ============================================================================

def sanitize_value(value: Any) -> Any:
    """Convert string 'None' to actual None"""
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return value

def normalize_name(name: Optional[str]) -> Optional[str]:
    """Normalize name for comparison"""
    if not name:
        return None
    name = sanitize_value(name)
    if not name:
        return None
    return re.sub(r'\s+', ' ', name.strip().upper())

def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string in DD/MM/YYYY format"""
    if not date_str:
        return None
    date_str = sanitize_value(date_str)
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except:
        return None

def calculate_age(dob_str: Optional[str]) -> Optional[int]:
    """Calculate age from DOB string"""
    if not dob_str:
        return None
    dob_str = sanitize_value(dob_str)
    if not dob_str:
        return None
    dob = parse_date(dob_str)
    if dob:
        today = datetime.now()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    return None

def validate_pan_format(pan: Optional[str]) -> bool:
    """Validate PAN format: 5 letters + 4 digits + 1 letter"""
    if not pan:
        return False
    pan = sanitize_value(pan)
    if not pan:
        return False
    pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]$'
    return bool(re.match(pattern, pan.upper()))

def validate_aadhaar_format(aadhaar: Optional[str]) -> bool:
    """Validate Aadhaar format: 12 digits"""
    if not aadhaar:
        return False
    aadhaar = sanitize_value(aadhaar)
    if not aadhaar:
        return False
    cleaned = aadhaar.replace(" ", "")
    return bool(re.match(r'^\d{12}$', cleaned))

def extract_filing_timeliness(filed_u_s: Optional[str]) -> str:
    """Extract filing timeliness from Filed_u_s field"""
    if not filed_u_s:
        return "Not found in ITR document"
    
    filed_u_s = sanitize_value(filed_u_s)
    if not filed_u_s:
        return "Not found in ITR document"
    
    filed_u_s_upper = filed_u_s.upper()
    
    if "139" in filed_u_s_upper and "1" in filed_u_s_upper:
        if "(1)" in filed_u_s_upper or "( 1 )" in filed_u_s_upper:
            return "On time"
    
    if "139" in filed_u_s_upper and "4" in filed_u_s_upper:
        if "(4)" in filed_u_s_upper or "( 4 )" in filed_u_s_upper:
            return "Belated"
    
    if "139" in filed_u_s_upper and "5" in filed_u_s_upper:
        if "(5)" in filed_u_s_upper or "( 5 )" in filed_u_s_upper:
            return "Revised"
    
    if "139" in filed_u_s_upper and "8" in filed_u_s_upper:
        if "(8)" in filed_u_s_upper or "( 8 )" in filed_u_s_upper:
            return "Updated"
    
    if "139" in filed_u_s_upper and "9" in filed_u_s_upper:
        if "(9)" in filed_u_s_upper or "( 9 )" in filed_u_s_upper:
            return "Updated"
    
    return "Unknown"

def resolve_field_value(field_name: str, *values) -> Dict[str, Any]:
    """
    Resolve field value from multiple sources, track where it came from
    Returns: dict with 'value', 'source', and 'status'
    """
    sources = ['PAN', 'Aadhaar', 'ITR', 'Additional Details']
    
    for idx, val in enumerate(values):
        val = sanitize_value(val)
        if val is not None and val != "":
            source = sources[idx] if idx < len(sources) else f"Source {idx+1}"
            return {
                "value": val,
                "source": source,
                "status": "found"
            }
    
    return {
        "value": None,
        "source": None,
        "status": "not_found_in_any_document"
    }

def cross_verify_documents(pan_data: PANCardInput, 
                          aadhaar_data: AadhaarCardInput, 
                          itr_data: ITRDocumentInput) -> Dict[str, Any]:
    """Cross-verify all three documents and return verification results"""
    
    verification_results = {
        "is_verified": True,
        "mismatches": [],
        "warnings": [],
        "missing_fields": [],
        "verification_details": {}
    }
    
    # PAN Number Validation and Cross-verification
    pan_sources = {
        "pan_card": sanitize_value(pan_data.pan_number),
        "itr_document": sanitize_value(itr_data.PAN)
    }
    
    available_pans = {k: v for k, v in pan_sources.items() if v}
    
    if not available_pans:
        verification_results["missing_fields"].append("PAN number not found in any document")
        verification_results["is_verified"] = False
    else:
        # Validate format
        for source, pan in available_pans.items():
            if not validate_pan_format(pan):
                verification_results["mismatches"].append(f"Invalid PAN format in {source}: {pan}")
                verification_results["is_verified"] = False
        
        # Check if PANs match across documents
        if len(set(p.upper() for p in available_pans.values())) > 1:
            verification_results["mismatches"].append(
                f"PAN mismatch across documents: {pan_sources}"
            )
            verification_results["is_verified"] = False
    
    # Aadhaar Number Validation
    aadhaar_num = sanitize_value(aadhaar_data.aadhaar_number)
    if aadhaar_num:
        if not validate_aadhaar_format(aadhaar_num):
            verification_results["mismatches"].append(
                f"Invalid Aadhaar format: {aadhaar_num}"
            )
            verification_results["is_verified"] = False
    else:
        verification_results["missing_fields"].append("Aadhaar number not found in document")
        verification_results["is_verified"] = False
    
    # Name Cross-verification
    name_sources = {
        "pan_card": sanitize_value(pan_data.name),
        "aadhaar_card": sanitize_value(aadhaar_data.name),
        "itr_document": sanitize_value(itr_data.Name)
    }
    
    available_names = {k: normalize_name(v) for k, v in name_sources.items() if v}
    
    if not available_names:
        verification_results["missing_fields"].append("Name not found in any document")
        verification_results["is_verified"] = False
    else:
        verification_results["verification_details"]["normalized_names"] = available_names
        
        # Compare names pairwise
        from difflib import SequenceMatcher
        NAME_THRESHOLD = 0.9
        
        name_list = list(available_names.items())
        for i in range(len(name_list)):
            for j in range(i + 1, len(name_list)):
                source1, name1 = name_list[i]
                source2, name2 = name_list[j]
                similarity = SequenceMatcher(None, name1, name2).ratio()
                
                verification_results["verification_details"][f"name_similarity_{source1}_{source2}"] = round(similarity, 3)
                
                if similarity < NAME_THRESHOLD:
                    verification_results["mismatches"].append(
                        f"Name mismatch: {source1} ({name_sources[source1]}) vs {source2} ({name_sources[source2]}) - Similarity: {round(similarity, 3)}"
                    )
                    verification_results["is_verified"] = False
    
    # DOB Cross-verification
    dob_sources = {
        "pan_card": sanitize_value(pan_data.dob),
        "aadhaar_card": sanitize_value(aadhaar_data.date_of_birth)
    }
    
    available_dobs = {k: parse_date(v) for k, v in dob_sources.items() if v}
    
    if not available_dobs:
        verification_results["missing_fields"].append("Date of birth not found in any document")
        verification_results["warnings"].append("Cannot verify age without DOB")
    elif len(set(str(d) for d in available_dobs.values())) > 1:
        verification_results["mismatches"].append(
            f"DOB mismatch across documents: {dob_sources}"
        )
        verification_results["is_verified"] = False
    
    # Gender verification
    if not aadhaar_data.gender:
        verification_results["missing_fields"].append("Gender not found in Aadhaar document")
    
    return verification_results

def parse_numeric_value(value: Optional[str]) -> Optional[float]:
    """Parse numeric value from string (remove commas, handle empty)"""
    if not value:
        return None
    value = sanitize_value(value)
    if not value:
        return None
    try:
        cleaned = value.replace(",", "").strip()
        return float(cleaned) if cleaned else None
    except:
        return None


# ============================================================================
# PDF TO IMAGE CONVERSION
# ============================================================================

def convert_pdf_to_image(file_bytes: bytes) -> Image.Image:
    """Convert PDF to image (first page only)"""
    if convert_from_bytes is None:
        raise ImportError(
            "pdf2image is not installed. Please install it with: pip install pdf2image"
        )
    try:
        images = convert_from_bytes(file_bytes)
        if images:
            return images[0]  # Return first page
        else:
            raise ValueError("PDF conversion resulted in no images")
    except Exception:
        raise


def process_uploaded_file_sync(file_path: str) -> Image.Image:
    """
    Process a file from path - convert PDF to image if needed, otherwise return image.
    Synchronous version for streaming pipeline.
    Preprocesses image to avoid OCR resizing warnings.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        contents = f.read()
    
    # Check if PDF
    if file_path.lower().endswith('.pdf'):
        image = convert_pdf_to_image(contents)
    else:
        # It's already an image
        image = Image.open(BytesIO(contents))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess to avoid OCR resizing warnings
    image = preprocess_image_for_ocr(image)
    
    return image


async def process_uploaded_file(file: UploadFile) -> Image.Image:
    """Process uploaded file - convert PDF to image if needed, otherwise return image (async for FastAPI)"""
    contents = await file.read()
    
    # Check if PDF
    if file.content_type == 'application/pdf' or file.filename.lower().endswith('.pdf'):
        image = convert_pdf_to_image(contents)
    else:
        # It's already an image
        image = Image.open(BytesIO(contents))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess to avoid OCR resizing warnings
    image = preprocess_image_for_ocr(image)
    
    return image


# ============================================================================
# VIDEO VERIFICATION MODULE
# ============================================================================


class VideoVerificationService:
    """Encapsulates selfie-video verification pipeline for PAN/Aadhaar matching."""

    def __init__(self):
        self.frame_sample_rate = 3
        self.max_frames = 120
        self.blinks_required = 1
        self.head_motion_required = 0.15
        self.verif_cosine_thresh_id = 0.55
        self.verif_cosine_thresh_video = 0.50

        self.device = None
        self.mtcnn = None
        self.face_encoder = None
        self.cascade = None
        self._init_lock = threading.Lock()

    def _ensure_dependencies_available(self) -> None:
        if VIDEO_MODULE_IMPORT_ERROR:
            raise RuntimeError(
                "Video verification dependencies are missing. "
                "Install the optional extras listed in requirements.txt to enable this feature."
            ) from VIDEO_MODULE_IMPORT_ERROR

    def _initialize_models(self) -> None:
        self._ensure_dependencies_available()
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with self._init_lock:
            if self.mtcnn is None:
                try:
                    self.mtcnn = MTCNN(keep_all=False, device=self.device, post_process=False)
                except Exception:
                    self.mtcnn = None

            if self.face_encoder is None:
                try:
                    self.face_encoder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                except Exception:
                    self.face_encoder = None

            if self.cascade is None and cv2 is not None:
                try:
                    cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )
                    self.cascade = cascade if not cascade.empty() else None
                except Exception:
                    self.cascade = None

    @staticmethod
    def _resize_keep_aspect(img: np.ndarray, size: Tuple[int, int] = (160, 160)) -> Optional[np.ndarray]:
        try:
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None

    def _to_tensor_and_normalize(self, img_rgb: np.ndarray) -> torch.Tensor:
        tensor = transforms.ToTensor()(img_rgb)
        tensor = (tensor - 0.5) * 2.0
        return tensor.unsqueeze(0).to(self.device)

    @staticmethod
    def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        a_norm = a_np / (np.linalg.norm(a_np) + 1e-8)
        b_norm = b_np / (np.linalg.norm(b_np) + 1e-8)
        return 1.0 - float(np.dot(a_norm, b_norm))

    def _detect_and_align_face(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        if cv2 is None:
            return None, None

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None, None

        if self.mtcnn is not None:
            try:
                boxes, _ = self.mtcnn.detect(frame_rgb)
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0].astype(int)
                    H, W = frame_rgb.shape[:2]
                    x1 = max(0, min(x1, W - 1))
                    x2 = max(0, min(x2, W))
                    y1 = max(0, min(y1, H - 1))
                    y2 = max(0, min(y2, H))
                    aligned = self.mtcnn(frame_rgb)
                    if aligned is not None:
                        if isinstance(aligned, torch.Tensor):
                            crop = aligned.permute(1, 2, 0).int().cpu().numpy().astype(np.uint8)
                        else:
                            crop = np.array(aligned)
                        return crop, [int(x1), int(y1), int(x2), int(y2)]
                    crop = frame_rgb[y1:y2, x1:x2].copy()
                    return crop, [int(x1), int(y1), int(x2), int(y2)]
            except Exception:
                pass

        if self.cascade is not None:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    crop_rgb = cv2.cvtColor(frame_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)
                    return crop_rgb, [int(x), int(y), int(x + w), int(y + h)]
            except Exception:
                pass

        return None, None

    def _get_embedding(self, face_rgb: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        if face_rgb is None or self.face_encoder is None:
            return None
        resized = self._resize_keep_aspect(face_rgb, (160, 160))
        if resized is None:
            return None
        try:
            tensor = self._to_tensor_and_normalize(resized)
            with torch.no_grad():
                emb = self.face_encoder(tensor).detach().cpu().squeeze(0)
            return emb / (emb.norm() + 1e-8)
        except Exception:
            return None

    def _extract_frames(self, video_path: str) -> List[Tuple[np.ndarray, float]]:
        frames: List[Tuple[np.ndarray, float]] = []
        if cv2 is None or not os.path.exists(video_path):
            return frames

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():  # Try alternate backends
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path, cv2.CAP_ANY)
        if not cap.isOpened():
            return frames

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if video_fps <= 0 or video_fps > 120:
            video_fps = 25.0

        step = max(1, int(round(video_fps / self.frame_sample_rate)))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0 and frame is not None and frame.size > 0:
                timestamp = idx / video_fps if video_fps else 0.0
                frames.append((frame.copy(), timestamp))
                if len(frames) >= self.max_frames:
                    break
            idx += 1

        cap.release()
        return frames


    @staticmethod
    def _detect_head_movement(bounding_boxes: List[Optional[List[int]]]) -> float:
        centers = []
        for box in bounding_boxes:
            if box is None:
                continue
            x1, y1, x2, y2 = box
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        if len(centers) < 3:
            return 0.0
        centers_arr = np.array(centers)
        movement_score = np.clip((np.var(centers_arr[:, 0]) + np.var(centers_arr[:, 1])) / 2000.0, 0.0, 1.0)
        return float(movement_score)

    def _detect_blinks_and_motion(self, face_crops: List[Optional[np.ndarray]]) -> Dict[str, float]:
        blink_count = 0
        motion_scores = []
        prev = None

        for crop in face_crops:
            if crop is None:
                motion_scores.append(0.0)
                prev = None
                continue
            if prev is not None:
                try:
                    fa = self._resize_keep_aspect(prev, (160, 160))
                    fb = self._resize_keep_aspect(crop, (160, 160))
                    if fa is not None and fb is not None:
                        diff = np.mean((fa.astype(np.float32) - fb.astype(np.float32)) ** 2) / (255.0 ** 2)
                        motion_scores.append(float(np.clip(diff * 20.0, 0.0, 1.0)))
                    else:
                        motion_scores.append(0.0)
                except Exception:
                    motion_scores.append(0.0)
            else:
                motion_scores.append(0.0)
            prev = crop

        avg_motion = float(np.mean(motion_scores)) if motion_scores else 0.0

        intensities: List[Optional[float]] = []
        for crop in face_crops:
            if crop is None:
                intensities.append(None)
                continue
            resized = self._resize_keep_aspect(crop, (160, 160))
            if resized is None:
                intensities.append(None)
                continue
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            h = gray.shape[0]
            eye_region = gray[int(h * 0.25):int(h * 0.50), :]
            intensities.append(float(eye_region.mean()))

        deltas = []
        for i in range(1, len(intensities)):
            if intensities[i] is None or intensities[i - 1] is None:
                deltas.append(0.0)
            else:
                deltas.append(intensities[i] - intensities[i - 1])

        for i in range(1, len(deltas)):
            if deltas[i - 1] < -10 and deltas[i] > 6:
                blink_count += 1

        return {"blink_count": blink_count, "avg_motion": avg_motion}

    @staticmethod
    def _verify_similarity(emb_a: Optional[torch.Tensor], emb_b: Optional[torch.Tensor], thresh: float) -> Tuple[bool, float]:
        if emb_a is None or emb_b is None:
            return False, 1.0
        dist = VideoVerificationService._cosine_distance(emb_a, emb_b)
        return dist < thresh, float(dist)

    def _encode_document_face(self, image_path: str, label: str) -> Tuple[Optional[torch.Tensor], List[str]]:
        notes: List[str] = []
        if cv2 is None:
            notes.append("OpenCV is not available; cannot process documents for video verification")
            return None, notes

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            notes.append(f"{label} image could not be loaded for video verification.")
            return None, notes

        face_rgb, _ = self._detect_and_align_face(image_bgr)
        if face_rgb is None:
            face_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            notes.append(f"No clear face detected on {label}; using entire image as fallback.")

        embedding = self._get_embedding(face_rgb)
        if embedding is None:
            notes.append(f"Could not compute embedding for {label} document.")
        else:
            notes.append(f"{label} face processed successfully.")
        return embedding, notes

    def _base_result(self) -> Dict[str, Any]:
        return {
            "aadhaar_pan_match": {
                "matched": False,
                "distance": None,
                "threshold": self.verif_cosine_thresh_id,
                "status": "not_evaluated"
            },
            "pan_video_match": {
                "matched": False,
                "distance": None,
                "threshold": self.verif_cosine_thresh_video,
                "status": "not_evaluated"
            },
            "liveness_check": {
                "passed": False,
                "blink_count": 0,
                "avg_motion": 0.0,
                "head_movement": 0.0,
                "thresholds": {
                    "blinks_required": self.blinks_required,
                    "head_motion_required": self.head_motion_required
                }
            },
            "final_decision": "undetermined",
            "notes": [],
            "detailed_scores": {},
            "summary": {
                "frames_processed": 0,
                "frames_with_face": 0,
                "face_detection_rate": 0.0
            }
        }

    def error_result(self, message: str) -> Dict[str, Any]:
        result = self._base_result()
        result["final_decision"] = "error_video_verification"
        result["notes"].append(message)
        return result

    def run_pipeline(self, aadhaar_path: str, pan_path: str, selfie_path: str) -> Dict[str, Any]:
        result = self._base_result()

        try:
            self._initialize_models()
        except Exception as exc:
            result["final_decision"] = "error_video_verification"
            result["notes"].append(str(exc))
            return result

        notes = result["notes"]

        aadhaar_emb, aadhaar_notes = self._encode_document_face(aadhaar_path, "Aadhaar")
        pan_emb, pan_notes = self._encode_document_face(pan_path, "PAN")
        notes.extend(aadhaar_notes + pan_notes)

        aadhaar_pan_same, aadhaar_pan_dist = self._verify_similarity(
            aadhaar_emb, pan_emb, self.verif_cosine_thresh_id
        )
        result["aadhaar_pan_match"].update({
            "matched": aadhaar_pan_same,
            "distance": float(np.round(aadhaar_pan_dist, 4)),
            "status": "pass" if aadhaar_pan_same else "fail"
        })
        result["detailed_scores"]["aadhaar_pan_distance"] = float(np.round(aadhaar_pan_dist, 4))
        notes.append(
            f"Aadhaar↔PAN distance: {aadhaar_pan_dist:.4f} (threshold {self.verif_cosine_thresh_id})"
        )

        if not aadhaar_pan_same:
            result["final_decision"] = "reject_aadhaar_pan_mismatch"
            return result

        frames = self._extract_frames(selfie_path)
        if not frames:
            result["final_decision"] = "reject_no_video_frames"
            notes.append("No frames could be extracted from selfie video.")
            return result

        face_crops: List[Optional[np.ndarray]] = []
        bounding_boxes: List[Optional[List[int]]] = []
        embeddings: List[Optional[torch.Tensor]] = []

        for frame_bgr, _ in frames:
            face_rgb, box = self._detect_and_align_face(frame_bgr)
            if face_rgb is not None and face_rgb.dtype != np.uint8:
                face_rgb = np.clip(face_rgb, 0, 255).astype(np.uint8)
            face_crops.append(face_rgb)
            bounding_boxes.append(box)
            embeddings.append(self._get_embedding(face_rgb))

        frames_with_face = len([c for c in face_crops if c is not None])
        detection_rate = frames_with_face / max(1, len(face_crops))
        result["summary"] = {
            "frames_processed": len(frames),
            "frames_with_face": frames_with_face,
            "face_detection_rate": float(np.round(detection_rate, 3))
        }

        motion_data = self._detect_blinks_and_motion(face_crops)
        head_movement = self._detect_head_movement(bounding_boxes)
        blink_count = motion_data["blink_count"]
        avg_motion = motion_data["avg_motion"]

        liveness_pass = (
            (blink_count >= self.blinks_required or avg_motion > 0.08)
            and head_movement > self.head_motion_required
        )
        result["liveness_check"].update({
            "passed": liveness_pass,
            "blink_count": int(blink_count),
            "avg_motion": float(np.round(avg_motion, 4)),
            "head_movement": float(np.round(head_movement, 4))
        })
        result["detailed_scores"].update({
            "blink_count": int(blink_count),
            "avg_motion": float(np.round(avg_motion, 4)),
            "head_movement": float(np.round(head_movement, 4))
        })

        if not liveness_pass:
            if blink_count < self.blinks_required and avg_motion <= 0.08:
                notes.append(f"Insufficient blinks ({blink_count}) and motion ({avg_motion:.3f}).")
            if head_movement <= self.head_motion_required:
                notes.append(f"Head movement below threshold ({head_movement:.3f}).")

        valid_embs = [emb for emb in embeddings if emb is not None]
        if not valid_embs or pan_emb is None:
            pan_video_same = False
            pan_video_dist = 1.0
            notes.append("Missing embeddings for PAN↔Video verification.")
        else:
            stacked = torch.stack(valid_embs, dim=0)
            avg_emb = stacked.mean(dim=0)
            avg_emb = avg_emb / (avg_emb.norm() + 1e-8)
            pan_video_same, pan_video_dist = self._verify_similarity(
                avg_emb, pan_emb, self.verif_cosine_thresh_video
            )

        result["pan_video_match"].update({
            "matched": pan_video_same,
            "distance": float(np.round(pan_video_dist, 4)),
            "status": "pass" if pan_video_same else "fail"
        })
        result["detailed_scores"]["pan_video_distance"] = float(np.round(pan_video_dist, 4))
        notes.append(
            f"PAN↔Video distance: {pan_video_dist:.4f} (threshold {self.verif_cosine_thresh_video})"
        )

        rejection_reasons = []
        if not result["aadhaar_pan_match"]["matched"]:
            rejection_reasons.append("aadhaar_pan_mismatch")
        if not result["pan_video_match"]["matched"]:
            rejection_reasons.append("face_mismatch_video")
        if not result["liveness_check"]["passed"]:
            rejection_reasons.append("failed_liveness")

        if rejection_reasons:
            result["final_decision"] = "reject_" + "_".join(rejection_reasons)
        else:
            result["final_decision"] = "accept"

        return result
