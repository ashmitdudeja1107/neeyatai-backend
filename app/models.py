from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ExecutionStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    EXECUTING = "executing"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"

class CodeExecutionRequest(BaseModel):
    prompt: str = Field(..., description="Natural language description of the code to generate")
    session_id: Optional[str] = Field(None, description="Session ID for context awareness")
    previous_code: Optional[str] = Field(None, description="Previously generated code for modification")
    modification_request: Optional[str] = Field(None, description="Specific modification requested")

class ExecutionAttempt(BaseModel):
    attempt_number: int
    generated_code: str
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.now)

class CodeExecutionResponse(BaseModel):
    session_id: str
    status: ExecutionStatus
    attempts: List[ExecutionAttempt]
    final_code: Optional[str] = None
    success: bool = False
    total_attempts: int = 0
    total_execution_time: float = 0.0
    error_message: Optional[str] = None
    prompt: str
    created_at: datetime = Field(default_factory=datetime.now)

class SessionContext(BaseModel):
    session_id: str
    conversation_history: List[Dict[str, str]] = []
    last_successful_code: Optional[str] = None
    last_prompt: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str = "1.0.0"
    daytona_connected: bool = False
    gemini_connected: bool = False