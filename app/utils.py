import uuid
import json
import hashlib
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def generate_session_id() -> str:
    return str(uuid.uuid4())

def generate_short_id() -> str:
    return str(uuid.uuid4())[:8]

def sanitize_code(code: str) -> str:
    if not code:
        return ""
    
    lines = [line.rstrip() for line in code.split('\n')]
    
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    return '\n'.join(lines)

def format_execution_time(seconds: float) -> str:
    if seconds < 0.001:
        return "< 1ms"
    elif seconds < 1:
        return f"{int(seconds * 1000)}ms"
    else:
        return f"{seconds:.2f}s"

def truncate_output(text: str, max_length: int = 1000) -> str:
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + f"\n... (truncated, {len(text) - max_length} more characters)"

def extract_error_info(stderr: str) -> Dict[str, Any]:
    if not stderr:
        return {"type": "unknown", "message": "", "line": None}
    
    error_patterns = [
        (r'(\w+Error): (.+)', 'python_error'),
        (r'File ".+", line (\d+), in .+', 'syntax_location'),
        (r'Traceback \(most recent call last\):', 'traceback')
    ]
    
    error_info = {
        "type": "unknown",
        "message": stderr.strip(),
        "line": None,
        "suggestions": []
    }
    
    for pattern, error_type in error_patterns:
        match = re.search(pattern, stderr)
        if match:
            if error_type == 'python_error':
                error_info["type"] = match.group(1)
                error_info["message"] = match.group(2)
            elif error_type == 'syntax_location':
                error_info["line"] = int(match.group(1))
    
    if "NameError" in error_info["type"]:
        error_info["suggestions"].append("Check variable names and imports")
    elif "SyntaxError" in error_info["type"]:
        error_info["suggestions"].append("Check syntax, parentheses, and indentation")
    elif "ImportError" in error_info["type"] or "ModuleNotFoundError" in error_info["type"]:
        error_info["suggestions"].append("Check if the module is available or use built-in alternatives")
    
    return error_info

def hash_code(code: str) -> str:
    return hashlib.md5(code.encode()).hexdigest()

def is_code_similar(code1: str, code2: str, threshold: float = 0.8) -> bool:
    if not code1 or not code2:
        return False
    
    lines1 = set(line.strip() for line in code1.split('\n') if line.strip())
    lines2 = set(line.strip() for line in code2.split('\n') if line.strip())
    
    if not lines1 or not lines2:
        return False
    
    intersection = len(lines1.intersection(lines2))
    union = len(lines1.union(lines2))
    
    similarity = intersection / union if union > 0 else 0
    return similarity >= threshold

def clean_ai_response(response: str) -> str:
    if not response:
        return ""
    
    prefixes_to_remove = [
        "Here's the Python code:",
        "Here's the code:",
        "The Python code is:",
        "```python",
        "```"
    ]
    
    cleaned = response
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
    
    code_end_patterns = [
        r'\n\nThis code.*',
        r'\n\nExplanation:.*',
        r'\n\nNote:.*'
    ]
    
    for pattern in code_end_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
    
    return cleaned.strip()

def validate_session_id(session_id: str) -> bool:
    if not session_id:
        return False
    
    try:
        uuid.UUID(session_id)
        return True
    except ValueError:
        return False

def create_execution_summary(attempts: List[Dict]) -> Dict[str, Any]:
    if not attempts:
        return {
            "total_attempts": 0,
            "success_rate": 0,
            "avg_execution_time": 0,
            "common_errors": []
        }
    
    total_attempts = len(attempts)
    successful_attempts = sum(1 for attempt in attempts if attempt.get("exit_code") == 0)
    success_rate = (successful_attempts / total_attempts) * 100
    
    total_time = sum(attempt.get("execution_time", 0) for attempt in attempts)
    avg_execution_time = total_time / total_attempts if total_attempts > 0 else 0
    
    error_types = {}
    for attempt in attempts:
        if attempt.get("exit_code") != 0 and attempt.get("stderr"):
            error_info = extract_error_info(attempt["stderr"])
            error_type = error_info["type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "total_attempts": total_attempts,
        "successful_attempts": successful_attempts,
        "success_rate": round(success_rate, 2),
        "avg_execution_time": round(avg_execution_time, 3),
        "total_execution_time": round(total_time, 3),
        "common_errors": common_errors[:3]
    }

def log_execution_metrics(session_id: str, attempts: List[Dict], success: bool):
    summary = create_execution_summary(attempts)
    
    logger.info(
        f"Session {session_id}: "
        f"Attempts={summary['total_attempts']}, "
        f"Success={'Yes' if success else 'No'}, "
        f"Time={summary['total_execution_time']}s, "
        f"Rate={summary['success_rate']}%"
    )

def safe_json_serialize(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, indent=2)
    except Exception as e:
        logger.error(f"JSON serialization failed: {str(e)}")
        return "{}"

def parse_modification_request(prompt: str, previous_code: str) -> Dict[str, Any]:
    modification_keywords = {
        'add': ['add', 'include', 'insert', 'append'],
        'remove': ['remove', 'delete', 'exclude', 'take out'],
        'change': ['change', 'modify', 'update', 'alter', 'replace'],
        'fix': ['fix', 'correct', 'repair', 'debug'],
        'optimize': ['optimize', 'improve', 'enhance', 'make better']
    }
    
    prompt_lower = prompt.lower()
    detected_intents = []
    
    for intent, keywords in modification_keywords.items():
        if any(keyword in prompt_lower for keyword in keywords):
            detected_intents.append(intent)
    
    return {
        "intents": detected_intents,
        "is_modification": bool(detected_intents),
        "has_previous_code": bool(previous_code and previous_code.strip()),
        "modification_type": detected_intents[0] if detected_intents else "unknown"
    }