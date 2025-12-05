"""
User Data Consumer with Pathway-Kafka Integration, Document Processing, and Risk Scoring.

Uses Pathway's Kafka connector to consume user data,
process documents (Aadhaar, PAN, ITR) using OCR,
perform video verification for liveness,
cross-verify documents,
calculate risk scores using the OnlineRiskScorer,
and assign users to one of 12 risk groups.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import base64
import tempfile
import os

import pathway as pw
from pathway.io import kafka as pw_kafka
from PIL import Image
from io import BytesIO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from streaming.config import (
    KAFKA_BROKER_USER,
    USER_DATA_TOPIC,
    KAFKA_GROUP_USER,
    RISK_ARTIFACTS_DIR,
    RISK_BOUNDARIES,
    NUM_RISK_GROUPS,
    USER_RISK_DATA_DIR,
)
from streaming.shared.state import SharedState
from streaming.shared.utils import decode_image_base64, decode_video_base64
from streaming.consumer.schemas import UserDataSchema, RiskScoredUserSchema

# Import KYC processing functions
from streaming.consumer.kyc_v import (
    parser_,
    PANCardInput,
    AadhaarCardInput,
    ITRDocumentInput,
    cross_verify_documents,
    VideoVerificationService,
    calculate_age,
    extract_filing_timeliness,
    parse_numeric_value,
    sanitize_value,
    Gender,
)

logger = logging.getLogger(__name__)

# Global risk scorer (loaded once)
_risk_scorer = None
_offline_kmeans = None
_cluster_context = None

# Global video verification service (loaded once)
_video_verifier = None


def get_risk_scorer():
    """Get or create the risk scorer instance."""
    global _risk_scorer, _offline_kmeans, _cluster_context
    if _risk_scorer is None:
        try:
            # Import risk scorer
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from investor_risk_scorer import load_deployment_pipeline
            
            artifacts_dir = str(RISK_ARTIFACTS_DIR)
            if RISK_ARTIFACTS_DIR.exists():
                _risk_scorer, _offline_kmeans, _cluster_context = load_deployment_pipeline(artifacts_dir)
                logger.info(f"Loaded risk scorer from {artifacts_dir}")
            else:
                logger.warning(f"Risk artifacts not found at {artifacts_dir}")
                _risk_scorer = None
        except Exception as e:
            logger.error(f"Failed to load risk scorer: {e}")
            _risk_scorer = None
    return _risk_scorer


def get_video_verifier():
    """Get or create the video verification service instance."""
    global _video_verifier
    if _video_verifier is None:
        try:
            _video_verifier = VideoVerificationService()
            logger.info("Initialized video verification service")
        except Exception as e:
            logger.error(f"Failed to initialize video verifier: {e}")
            _video_verifier = None
    return _video_verifier


def get_risk_group(score: float) -> int:
    """
    Map a risk score (0-100) to one of 12 risk groups.
    
    Returns group index 0-11.
    """
    for idx, (low, high) in enumerate(RISK_BOUNDARIES):
        if low <= score < high:
            return idx
    return NUM_RISK_GROUPS - 1  # Last group for score == 100


def get_risk_label(score: float) -> str:
    """Get risk label based on score."""
    if score < 33.33:
        return "Conservative"
    elif score < 66.67:
        return "Moderate"
    else:
        return "Aggressive"


@pw.udf
def process_documents(
    userid: int,
    aadhar_base64: str,
    pan_base64: str,
    itr_base64: str,
    video_path: str,
    main_occupation: str,
    marital_status: str,
    dependents: int,
    q1: str,
    q2: str,
    q3: str,
    q4: str,
    q5: str,
    q6: str,
) -> str:
    """
    Process documents using OCR, perform video verification, and cross-verify.
    Returns JSON string with extracted data and verification results.
    """
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        
        # Initialize result structure
        result = {
            "userid": userid,
            "parsed_data": {
                "pan": {},
                "aadhaar": {},
                "itr": {}
            },
            "extracted_fields": {
                "age": None,
                "gender": None,
                "gross_income": None,
                "tax_paid": None,
                "form_type": None,
                "filing_timeliness": None,
                "main_occupation": main_occupation,
                "marital_status": marital_status,
                "dependents": dependents,
            },
            "verification": {
                "document_verification": {},
                "video_verification": {},
                "is_verified": False,
            },
            "questionnaire": {
                "Q1": q1, "Q2": q2, "Q3": q3, 
                "Q4": q4, "Q5": q5, "Q6": q6
            },
            "errors": []
        }
        
        # Decode and save documents to temp files
        pan_path = None
        aadhaar_path = None
        itr_path = None
        # Note: video_path comes from function parameter, don't overwrite it
        
        # Decode PAN
        if pan_base64:
            try:
                pan_bytes = base64.b64decode(pan_base64)
                pan_path = os.path.join(temp_dir, "pan.png")
                with open(pan_path, "wb") as f:
                    f.write(pan_bytes)
            except Exception as e:
                result["errors"].append(f"PAN decode error: {str(e)}")
        
        # Decode Aadhaar
        if aadhar_base64:
            try:
                aadhaar_bytes = base64.b64decode(aadhar_base64)
                aadhaar_path = os.path.join(temp_dir, "aadhaar.png")
                with open(aadhaar_path, "wb") as f:
                    f.write(aadhaar_bytes)
            except Exception as e:
                result["errors"].append(f"Aadhaar decode error: {str(e)}")
        
        # Decode ITR
        if itr_base64:
            try:
                itr_bytes = base64.b64decode(itr_base64)
                itr_path = os.path.join(temp_dir, "itr.png")
                with open(itr_path, "wb") as f:
                    f.write(itr_bytes)
            except Exception as e:
                result["errors"].append(f"ITR decode error: {str(e)}")
        
        # Resolve video path from shared volume (not base64 encoded due to size)
        resolved_video_path = None
        if video_path:
            # Import SMARTFOLIO_DIR for path resolution
            try:
                from streaming.config import SMARTFOLIO_DIR
                smartfolio_dir = str(SMARTFOLIO_DIR)
            except ImportError:
                smartfolio_dir = str(Path(__file__).resolve().parents[2])
            
            # Try Docker path first, then local path
            if video_path.startswith("/data/"):
                docker_path = "/app" + video_path  # /app/data/...
                local_path = smartfolio_dir + video_path
                if os.path.exists(docker_path):
                    resolved_video_path = docker_path
                elif os.path.exists(local_path):
                    resolved_video_path = local_path
            elif os.path.exists(video_path):
                resolved_video_path = video_path
            
            if not resolved_video_path:
                result["errors"].append(f"Video file not found: {video_path}")
        
        # Parse PAN
        pan_data_dict = {}
        if pan_path and os.path.exists(pan_path):
            try:
                pan_image = Image.open(pan_path)
                if pan_image.mode != 'RGB':
                    pan_image = pan_image.convert('RGB')
                pan_data_dict = parser_(pan_image, 'pan')
                result["parsed_data"]["pan"] = pan_data_dict
            except Exception as e:
                result["errors"].append(f"PAN parsing error: {str(e)}")
        
        # Parse Aadhaar
        aadhaar_data_dict = {}
        if aadhaar_path and os.path.exists(aadhaar_path):
            try:
                aadhaar_image = Image.open(aadhaar_path)
                if aadhaar_image.mode != 'RGB':
                    aadhaar_image = aadhaar_image.convert('RGB')
                aadhaar_data_dict = parser_(aadhaar_image, 'adhaar')
                result["parsed_data"]["aadhaar"] = aadhaar_data_dict
            except Exception as e:
                result["errors"].append(f"Aadhaar parsing error: {str(e)}")
        
        # Parse ITR
        itr_data_dict = {}
        if itr_path and os.path.exists(itr_path):
            try:
                itr_image = Image.open(itr_path)
                if itr_image.mode != 'RGB':
                    itr_image = itr_image.convert('RGB')
                itr_data_dict = parser_(itr_image, 'itr')
                result["parsed_data"]["itr"] = itr_data_dict
            except Exception as e:
                result["errors"].append(f"ITR parsing error: {str(e)}")
        
        # Extract fields from parsed data
        # Age from DOB (PAN or Aadhaar)
        dob = pan_data_dict.get("dob") or aadhaar_data_dict.get("date_of_birth")
        if dob:
            age = calculate_age(dob)
            result["extracted_fields"]["age"] = age
        
        # Gender from Aadhaar
        gender = aadhaar_data_dict.get("gender")
        if gender:
            result["extracted_fields"]["gender"] = gender
        
        # Financial data from ITR
        gross_income = parse_numeric_value(itr_data_dict.get("Total_Income"))
        tax_paid = parse_numeric_value(itr_data_dict.get("Taxes_Paid"))
        form_type = sanitize_value(itr_data_dict.get("ITR_Type"))
        filing_timeliness = extract_filing_timeliness(itr_data_dict.get("Filed_u_s"))
        
        result["extracted_fields"]["gross_income"] = gross_income
        result["extracted_fields"]["tax_paid"] = tax_paid
        result["extracted_fields"]["form_type"] = form_type
        result["extracted_fields"]["filing_timeliness"] = filing_timeliness
        
        # Cross-verify documents
        try:
            # Convert gender string to Gender enum if needed
            gender_enum = None
            if gender:
                gender_lower = gender.lower()
                if gender_lower == "male":
                    gender_enum = Gender.MALE
                elif gender_lower == "female":
                    gender_enum = Gender.FEMALE
                elif gender_lower == "transgender":
                    gender_enum = Gender.TRANSGENDER
                else:
                    gender_enum = Gender.OTHER
            
            pan_input = PANCardInput(**pan_data_dict)
            aadhaar_input = AadhaarCardInput(
                aadhaar_number=aadhaar_data_dict.get("aadhaar_number"),
                name=aadhaar_data_dict.get("name"),
                date_of_birth=aadhaar_data_dict.get("date_of_birth"),
                gender=gender_enum
            )
            itr_input = ITRDocumentInput(**itr_data_dict)
            
            doc_verification = cross_verify_documents(pan_input, aadhaar_input, itr_input)
            result["verification"]["document_verification"] = doc_verification
        except Exception as e:
            result["errors"].append(f"Cross-verification error: {str(e)}")
            result["verification"]["document_verification"] = {
                "is_verified": False,
                "error": str(e)
            }
        
        # Video verification
        if resolved_video_path and aadhaar_path and pan_path:
            try:
                verifier = get_video_verifier()
                if verifier:
                    video_result = verifier.run_pipeline(aadhaar_path, pan_path, resolved_video_path)
                    result["verification"]["video_verification"] = video_result
                else:
                    result["verification"]["video_verification"] = {
                        "final_decision": "error_video_verification",
                        "notes": ["Video verifier not available"]
                    }
            except Exception as e:
                result["errors"].append(f"Video verification error: {str(e)}")
                result["verification"]["video_verification"] = {
                    "final_decision": "error_video_verification",
                    "notes": [str(e)]
                }
        else:
            result["verification"]["video_verification"] = {
                "final_decision": "skipped",
                "notes": ["Missing required documents for video verification"]
            }
        
        # Determine overall verification status
        doc_verified = result["verification"]["document_verification"].get("is_verified", False)
        video_decision = result["verification"]["video_verification"].get("final_decision", "")
        video_verified = video_decision == "accept"
        
        result["verification"]["is_verified"] = doc_verified and video_verified
        
        # Log processing summary
        print(f"[User {userid}] Processed documents:")
        print(f"  - PAN parsed: {bool(pan_data_dict)}")
        print(f"  - Aadhaar parsed: {bool(aadhaar_data_dict)}")
        print(f"  - ITR parsed: {bool(itr_data_dict)}")
        print(f"  - Video path resolved: {resolved_video_path}")
        print(f"  - Doc verified: {doc_verified}, Video verified: {video_verified}")
        if result["errors"]:
            print(f"  - Errors: {result['errors']}")
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error processing documents for user {userid}: {e}")
        return json.dumps({
            "userid": userid,
            "parsed_data": {"pan": {}, "aadhaar": {}, "itr": {}},
            "extracted_fields": {
                "age": None, "gender": None, "gross_income": None,
                "tax_paid": None, "form_type": None, "filing_timeliness": None,
                "main_occupation": main_occupation, "marital_status": marital_status,
                "dependents": dependents,
            },
            "verification": {"is_verified": False, "error": str(e)},
            "questionnaire": {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q6": q6},
            "errors": [str(e)]
        })
    finally:
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


@pw.udf
def calculate_risk_score(
    userid: int,
    processed_data_json: str,
) -> str:
    """
    Calculate risk score for a user based on processed document data.
    Returns JSON string with score, label, and group.
    """
    try:
        # Parse processed data
        processed_data = json.loads(processed_data_json)
        extracted = processed_data.get("extracted_fields", {})
        questionnaire = processed_data.get("questionnaire", {})
        
        # Get extracted fields
        age = extracted.get("age")
        gender = extracted.get("gender", "")
        main_occupation = extracted.get("main_occupation", "")
        marital_status = extracted.get("marital_status", "")
        dependents = extracted.get("dependents", 0)
        gross_income = extracted.get("gross_income")
        tax_paid = extracted.get("tax_paid")
        form_type = extracted.get("form_type", "")
        filing_timeliness = extracted.get("filing_timeliness", "")
        
        # Get questionnaire
        q1 = questionnaire.get("Q1", "")
        q2 = questionnaire.get("Q2", "")
        q3 = questionnaire.get("Q3", "")
        q4 = questionnaire.get("Q4", "")
        q5 = questionnaire.get("Q5", "")
        q6 = questionnaire.get("Q6", "")
        
        scorer = get_risk_scorer()
        
        if scorer is None:
            print("chutiyapa done")
            # Fallback: calculate simple score based on available data
            score = calculate_fallback_score(
                age or 35,  # Default age if not extracted
                dependents or 0,
                gross_income or 0,
                q1, q2, q3, q4, q5, q6
            )
        else:
            print("DONE")
            # Prepare user data for scoring
            user_data = {
                "age": age,
                "gender": gender,
                "main_occupation": main_occupation,
                "marital_status": marital_status,
                "dependents": dependents,
                "gross_income": gross_income,
                "tax_paid": tax_paid,
                "form_type": form_type,
                "filing_timeliness": filing_timeliness,
                "Q1": q1,
                "Q2": q2,
                "Q3": q3,
                "Q4": q4,
                "Q5": q5,
                "Q6": q6,
            }
            
            # Score the user using the pipeline
            try:
                result = scorer.predict_and_update(user_data, update=False)
                score = result.get("risk_score", 50.0)
            except Exception as e:
                logger.warning(f"Scorer failed, using fallback: {e}")
                score = calculate_fallback_score(
                    age or 35, dependents or 0, gross_income or 0,
                    q1, q2, q3, q4, q5, q6
                )
        
        # Calculate group and label
        risk_group = get_risk_group(score)
        risk_label = get_risk_label(score)
        
        # Get verification status
        verification = processed_data.get("verification", {})
        is_verified = verification.get("is_verified", False)
        
        result = {
            "userid": userid,
            "score": round(score, 2),
            "label": risk_label,
            "group": risk_group,
            "is_verified": is_verified,
            "extracted_data": extracted,
            "verification_summary": {
                "documents_verified": verification.get("document_verification", {}).get("is_verified", False),
                "video_verified": verification.get("video_verification", {}).get("final_decision", "") == "accept",
            }
        }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error calculating risk score: {e}")
        return json.dumps({
            "userid": userid,
            "score": 50.0,  # Default middle score
            "label": "Moderate",
            "group": 5,
            "is_verified": False,
            "error": str(e),
        })


def calculate_fallback_score(
    age: int,
    dependents: int,
    gross_income: float,
    q1: str, q2: str, q3: str, q4: str, q5: str, q6: str
) -> float:
    """
    Calculate a simple fallback risk score when model is not available.
    
    Higher score = more risk-tolerant (aggressive)
    Lower score = less risk-tolerant (conservative)
    """
    score = 50.0  # Start at middle
    
    # Age factor (younger = more risk tolerant)
    if age < 30:
        score += 15
    elif age < 45:
        score += 5
    elif age > 55:
        score -= 10
    
    # Dependents (more dependents = more conservative)
    score -= dependents * 3
    
    # Income (higher income = can afford more risk)
    if gross_income > 200000:
        score += 10
    elif gross_income > 100000:
        score += 5
    elif gross_income < 50000:
        score -= 5
    
    # Questionnaire responses (simplified scoring)
    q_responses = [q1, q2, q3, q4, q5, q6]
    for q in q_responses:
        if q and len(q) > 0:
            # Assume higher letter = more aggressive
            if q.upper() in ['D', 'E', '4', '5']:
                score += 3
            elif q.upper() in ['A', 'B', '1', '2']:
                score -= 2
    
    # Clamp to valid range
    return max(0, min(100, score))


@pw.udf
def persist_user_risk(risk_result_json: str, processed_data_json: str) -> str:
    """
    Persist user risk data, verification results, and add to appropriate group.
    Returns status message.
    """
    try:
        result = json.loads(risk_result_json)
        processed_data = json.loads(processed_data_json)
        
        userid = result["userid"]
        score = result["score"]
        label = result["label"]
        group = result["group"]
        is_verified = result.get("is_verified", False)
        
        # Get shared state
        state = SharedState()
        
        # Add to user risk data
        state.add_user_risk_data(userid, score, label, group)
        
        # Save to file with comprehensive data
        USER_RISK_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_file = USER_RISK_DATA_DIR / f"user_{userid}.json"
        
        # Get full verification details from processed_data
        full_verification = processed_data.get("verification", {})
        
        output_data = {
            "userid": userid,
            "risk_score": score,
            "risk_label": label,
            "risk_group": group,
            "is_verified": is_verified,
            "verification": {
                "documents_verified": full_verification.get("document_verification", {}).get("is_verified", False),
                "video_verified": full_verification.get("video_verification", {}).get("final_decision", "") == "accept",
                # Include full document verification details
                "document_verification_details": full_verification.get("document_verification", {}),
                # Include full video verification details from kyc_v.py
                "video_verification_details": full_verification.get("video_verification", {}),
            },
            "extracted_data": result.get("extracted_data", {}),
            "parsed_documents": processed_data.get("parsed_data", {}),
            "questionnaire": processed_data.get("questionnaire", {}),
            "errors": processed_data.get("errors", []),
            "processed_at": datetime.now().isoformat(),
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(
            f"User {userid}: score={score:.2f}, label={label}, group={group}, verified={is_verified}"
        )
        
        return f"Persisted user {userid} to group {group} (verified={is_verified})"
        
    except Exception as e:
        logger.error(f"Error persisting user risk: {e}")
        return f"Error: {str(e)}"


class UserConsumer:
    """
    Pathway-based Kafka consumer for user data with document processing and risk scoring.
    
    Consumes from user_data topic, processes documents (Aadhaar, PAN, ITR),
    performs video verification, cross-verifies documents,
    calculates risk scores, assigns users to risk groups, and persists the data.
    """
    
    def __init__(
        self,
        broker: str = KAFKA_BROKER_USER,
        topic: str = USER_DATA_TOPIC,
        group_id: str = KAFKA_GROUP_USER,
    ):
        self.broker = broker
        self.topic = topic
        self.group_id = group_id
        self._state = SharedState()
    
    def build_pipeline(self) -> pw.Table:
        """
        Build the Pathway pipeline for user data consumption.
        
        Returns:
            Pathway Table with risk-scored user data
        """
        # Kafka input connector
        logger.info(f"Setting up Kafka consumer for topic {self.topic}")
        input_table = pw_kafka.read(
            rdkafka_settings={
                "bootstrap.servers": self.broker,
                "group.id": self.group_id,
                "auto.offset.reset": "earliest",
            },
            topic=self.topic,
            format="json",
            schema=UserDataSchema,
        )
        logger.info(f"{input_table}")
        # Step 1: Process documents (OCR, video verification, cross-verification)
        with_processed = input_table.select(
            userid=pw.this.userid,
            first=pw.this.first,
            last=pw.this.last,
            processed_data=process_documents(
                pw.this.userid,
                pw.this.aadhar_base64,
                pw.this.pan_base64,
                pw.this.itr_base64,
                pw.this.video_path,
                pw.this.main_occupation,
                pw.this.marital_status,
                pw.this.dependents,
                pw.this.Q1,
                pw.this.Q2,
                pw.this.Q3,
                pw.this.Q4,
                pw.this.Q5,
                pw.this.Q6,
            ),
        )
        # Step 2: Calculate risk scores based on processed data
        with_risk = with_processed.select(
            userid=pw.this.userid,
            first=pw.this.first,
            last=pw.this.last,
            processed_data=pw.this.processed_data,
            risk_result=calculate_risk_score(
                pw.this.userid,
                pw.this.processed_data,
            ),
        )
        
        # Step 3: Persist user risk data
        processed = with_risk.select(
            userid=pw.this.userid,
            first=pw.this.first,
            last=pw.this.last,
            risk_result=pw.this.risk_result,
            processed_data=pw.this.processed_data,
            status=persist_user_risk(pw.this.risk_result, pw.this.processed_data),
        )
        
        return processed
    
    def run(self):
        """Start the Pathway runtime for user consumption."""
        logger.info(f"Starting User Consumer on {self.topic}")
        
        # Build pipeline
        processed = self.build_pipeline()
        
        # Suppress output
        pw.io.null.write(processed)
        
        # Run Pathway without monitoring dashboard
        pw.run(monitoring_level=pw.MonitoringLevel.NONE)


def create_user_consumer_pipeline() -> pw.Table:
    """
    Create and return the user consumer pipeline.
    """
    consumer = UserConsumer()
    return consumer.build_pipeline()


def run_user_consumer():
    """Main entry point for user consumer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.info("Starting User Consumer...")
    consumer = UserConsumer()
    consumer.run()


if __name__ == "__main__":
    run_user_consumer()
