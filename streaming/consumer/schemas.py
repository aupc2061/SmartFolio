"""
Pathway Schemas for Kafka Message Parsing.

Defines the schema structures for:
- StockDataSchema: OHLCV stock data messages
- UserDataSchema: User data with encoded documents
"""

import pathway as pw


class StockDataSchema(pw.Schema):
    """
    Schema for stock OHLCV data messages from Kafka.
    
    Each message contains:
    - date: Trading date (YYYY-MM-DD format)
    - data: JSON string with ticker->OHLCV mapping
    """
    date: str
    data: str  # JSON string with {ticker: {Open, High, Low, Close, Volume}}


class UserDataSchema(pw.Schema):
    """
    Schema for user data messages from Kafka.
    
    Contains user profile with document paths and base64 encoded documents.
    Note: Using 'userid' instead of 'id' (reserved in Pathway)
    """
    userid: int
    first: str
    last: str
    # Document paths (for reference)
    aadhar_path: str
    pan_path: str
    itr_path: str
    video_path: str
    # Base64 encoded documents
    aadhar_base64: str
    pan_base64: str
    itr_base64: str
    video_base64: str
    # Additional details from CSV
    main_occupation: str
    marital_status: str
    dependents: int
    # Questionnaire
    Q1: str
    Q2: str
    Q3: str
    Q4: str
    Q5: str
    Q6: str


class RiskScoredUserSchema(pw.Schema):
    """
    Schema for user data after risk scoring.
    """
    userid: int
    first: str
    last: str
    risk_score: float
    risk_label: str  # Conservative, Moderate, Aggressive
    risk_group: int  # 0-11 (one of 12 groups)
    processed_at: str


class MonthlyStockSummarySchema(pw.Schema):
    """
    Schema for monthly stock data summaries.
    """
    year_month: str  # YYYY-MM format
    row_count: int
    saved_path: str
    processed_at: str
