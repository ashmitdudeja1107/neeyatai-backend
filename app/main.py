from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import asyncio
from typing import Dict
from datetime import datetime

from app.config import settings
from app.models import (
    CodeExecutionRequest, CodeExecutionResponse, ExecutionAttempt,
    SessionContext, HealthResponse, ExecutionStatus
)
from app.ai_generator import AICodeGenerator
from app.sandbox import DaytonaSandbox
from app.validation import CodeValidator, validate_prompt
from app.utils import (
    generate_session_id, format_execution_time, extract_error_info,
    is_code_similar, log_execution_metrics, parse_modification_request
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Recursive AI Executor",
    description="Generate, execute, and refine Python code using AI with smart input handling",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_generator = AICodeGenerator()
sandbox = DaytonaSandbox()
validator = CodeValidator()
sessions: Dict[str, SessionContext] = {}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Enhanced Recursive AI Executor with Smart Input Handling")
    
    try:
        gemini_status = await ai_generator.check_connection()
        daytona_status = await sandbox._check_daytona_connection()
        
        logger.info(f"Gemini connection: {'OK' if gemini_status else 'FAILED'}")
        logger.info(f"Daytona connection: {'OK' if daytona_status else 'FAILED'}")
        logger.info("Smart input handling: ENABLED")
        
    except Exception as e:
        logger.error(f"Startup connection tests failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Enhanced Recursive AI Executor")
    
    try:
        await sandbox.close()
        logger.info("Daytona resources cleaned up")
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {str(e)}")

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Enhanced Recursive AI Executor API",
        "version": "1.0.0",
        "ai_model": "Google Gemini",
        "features": "Smart Input Handling, Auto-Retry, Context-Aware Execution",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        gemini_connected = await ai_generator.check_connection()
        daytona_connected = await sandbox._check_daytona_connection()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            gemini_connected=gemini_connected,
            daytona_connected=daytona_connected
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            gemini_connected=False,
            daytona_connected=False
        )

@app.post("/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    prompt_validation = validate_prompt(request.prompt)
    if not prompt_validation['is_valid']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid prompt: {', '.join(prompt_validation['errors'])}"
        )
    
    session_id = request.session_id or generate_session_id()
    
    if session_id not in sessions:
        sessions[session_id] = SessionContext(session_id=session_id)
    
    session = sessions[session_id]
    modification_info = parse_modification_request(request.prompt, request.previous_code)
    
    response = CodeExecutionResponse(
        session_id=session_id,
        status=ExecutionStatus.PENDING,
        attempts=[],
        prompt=request.prompt
    )
    
    try:
        response = await _recursive_execute_enhanced(
            request, session, response, modification_info
        )
        
        _update_session_context(session, request, response)
        
        log_execution_metrics(session_id, [
            {
                "exit_code": attempt.exit_code,
                "execution_time": attempt.execution_time,
                "stderr": attempt.stderr
            } for attempt in response.attempts
        ], response.success)
        
        return response
        
    except Exception as e:
        logger.error(f"Execution failed for session {session_id}: {str(e)}")
        response.status = ExecutionStatus.ERROR
        response.error_message = str(e)
        return response

async def _recursive_execute_enhanced(
    request: CodeExecutionRequest,
    session: SessionContext,
    response: CodeExecutionResponse,
    modification_info: Dict
) -> CodeExecutionResponse:
    current_code = request.previous_code
    last_error = None
    
    for attempt_num in range(1, settings.max_retry_attempts + 1):
        success_analysis = None
        execution_result = None
        stdout = ""
        stderr = ""
        exit_code = 1
        exec_time = 0.0
        validation_result = None
        
        try:
            response.status = ExecutionStatus.GENERATING
            logger.info(f"Starting attempt {attempt_num} for session {session.session_id}")
            
            generated_code = await ai_generator.generate_code(
                prompt=request.prompt,
                context=session,
                previous_code=current_code,
                error_output=last_error,
                attempt_number=attempt_num
            )
            
            try:
                validation_result = validator.validate_code(
                    code=generated_code, 
                    prompt=request.prompt
                )
                
                if not validation_result['is_valid']:
                    last_error = f"Code validation failed: {', '.join(validation_result['errors'])}"
                    logger.warning(f"Validation failed in attempt {attempt_num}: {last_error}")
                    continue
                
            except Exception as val_error:
                logger.error(f"Validation error in attempt {attempt_num}: {str(val_error)}")
                last_error = f"Code validation error: {str(val_error)}"
                continue
            
            code_to_execute = validation_result['sanitized_code']
            
            if current_code and is_code_similar(current_code, code_to_execute):
                logger.warning(f"Similar code detected in attempt {attempt_num}")
                try:
                    modified_prompt = f"{request.prompt}\n\nPlease try a different approach or method."
                    generated_code = await ai_generator.generate_code(
                        prompt=modified_prompt,
                        context=session,
                        previous_code=current_code,
                        error_output=last_error,
                        attempt_number=attempt_num
                    )
                    validation_result = validator.validate_code(generated_code, request.prompt)
                    code_to_execute = validation_result['sanitized_code']
                except Exception as similarity_error:
                    logger.error(f"Error handling similar code: {str(similarity_error)}")
            
            response.status = ExecutionStatus.EXECUTING
            
            logger.info(f"Executing code with smart input handling for attempt {attempt_num}")
            
            try:
                execution_result = validator.execute_code_with_smart_inputs(
                    code=code_to_execute, 
                    prompt=request.prompt
                )
                
                stdout = execution_result.get('stdout', '')
                stderr = execution_result.get('stderr', '')
                exit_code = execution_result.get('return_code', 1)
                exec_time = execution_result.get('execution_time', 0.0)
                
                execution_method = execution_result.get('execution_method', 'unknown')
                logger.info(f"Execution method used: {execution_method}")
                
                if execution_result.get('inputs_provided'):
                    logger.info(f"Auto-inputs provided: {execution_result['inputs_provided']}")
                
                if execution_result.get('modified_code'):
                    logger.info("Code was modified for input replacement")
                    
            except Exception as exec_error:
                logger.error(f"Smart execution failed in attempt {attempt_num}: {str(exec_error)}")
                stdout = ""
                stderr = f"Smart execution error: {str(exec_error)}"
                exit_code = 1
                exec_time = 0.0
                execution_result = {
                    'stdout': stdout,
                    'stderr': stderr,
                    'return_code': exit_code,
                    'execution_time': exec_time,
                    'execution_method': 'failed'
                }
            
            attempt = ExecutionAttempt(
                attempt_number=attempt_num,
                generated_code=code_to_execute,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                execution_time=exec_time
            )
            
            if hasattr(attempt, 'execution_metadata') and execution_result:
                attempt.execution_metadata = {
                    'execution_method': execution_result.get('execution_method', 'unknown'),
                    'inputs_provided': execution_result.get('inputs_provided', []),
                    'code_modified': execution_result.get('modified_code') is not None,
                    'validation_confidence': validation_result.get('confidence_score', 0.0) if validation_result else 0.0
                }
            
            response.attempts.append(attempt)
            response.total_attempts = attempt_num
            response.total_execution_time += exec_time
            
            if exit_code == 0:
                try:
                    success_analysis = validator.validate_execution_success_only(
                        code=code_to_execute,
                        prompt=request.prompt,
                        execution_result=execution_result
                    )
                    
                    if success_analysis['overall_success']:
                        response.status = ExecutionStatus.SUCCESS
                        response.success = True
                        response.final_code = code_to_execute
                        
                        logger.info(
                            f"Code execution successful after {attempt_num} attempts. "
                            f"Quality: {success_analysis['output_quality']}, "
                            f"Confidence: {success_analysis['confidence_score']:.2f}"
                        )
                        break
                    else:
                        basic_success = (
                            exit_code == 0 and 
                            stdout.strip() and 
                            len(stdout.strip()) > 0 and
                            not (stderr.strip() and 'error' in stderr.lower())
                        )
                        
                        if basic_success:
                            response.status = ExecutionStatus.SUCCESS
                            response.success = True
                            response.final_code = code_to_execute
                            logger.info(f"Code execution successful (basic criteria) after {attempt_num} attempts")
                            break
                        else:
                            logger.warning(
                                f"Code executed (exit_code=0) but failed success validation: "
                                f"{success_analysis.get('failure_reasons', ['Unknown reasons'])}"
                            )
                            last_error = f"Code executed but output quality insufficient: {', '.join(success_analysis.get('failure_reasons', ['Output validation failed'])[:2])}"
                            current_code = code_to_execute
                
                except Exception as success_error:
                    logger.error(f"Success analysis failed in attempt {attempt_num}: {str(success_error)}")
                    if exit_code == 0 and stdout.strip():
                        response.status = ExecutionStatus.SUCCESS
                        response.success = True
                        response.final_code = code_to_execute
                        logger.info(f"Code execution successful (fallback) after {attempt_num} attempts")
                        break
                    else:
                        last_error = f"Success analysis failed: {str(success_error)}"
                        current_code = code_to_execute
            else:
                try:
                    error_info = extract_error_info(stderr)
                    
                    try:
                        enhanced_feedback = validator.get_enhanced_failure_feedback(
                            prompt=request.prompt,
                            execution_result=execution_result,
                            success_analysis=success_analysis or {'failure_reasons': [error_info.get('message', 'Unknown error')]}
                        )
                        last_error = enhanced_feedback
                    except Exception as feedback_error:
                        logger.error(f"Enhanced feedback failed: {str(feedback_error)}")
                        last_error = f"{stderr}\n\nError type: {error_info.get('type', 'Unknown')}"
                    
                    current_code = code_to_execute
                    logger.warning(f"Attempt {attempt_num} failed: {error_info.get('type', 'Unknown error')}")
                    
                except Exception as error_processing_error:
                    logger.error(f"Error processing failure in attempt {attempt_num}: {str(error_processing_error)}")
                    last_error = stderr or "Unknown execution error"
                    current_code = code_to_execute
                
                if attempt_num == settings.max_retry_attempts:
                    response.status = ExecutionStatus.ERROR
                    response.error_message = f"Max attempts reached. Last error: {last_error}"
        
        except Exception as e:
            logger.error(f"Critical error in attempt {attempt_num}: {str(e)}")
            
            attempt = ExecutionAttempt(
                attempt_number=attempt_num,
                generated_code=current_code or "",
                stdout="",
                stderr=str(e),
                exit_code=1,
                execution_time=0.0
            )
            
            response.attempts.append(attempt)
            response.total_attempts = attempt_num
            
            last_error = str(e)
            
            if attempt_num == settings.max_retry_attempts:
                response.status = ExecutionStatus.ERROR
                response.error_message = f"Execution failed: {str(e)}"
    
    return response

def _update_session_context(
    session: SessionContext,
    request: CodeExecutionRequest,
    response: CodeExecutionResponse
):
    session.conversation_history.append({
        "prompt": request.prompt,
        "code": response.final_code or "",
        "success": response.success,
        "attempts": response.total_attempts,
        "timestamp": datetime.now().isoformat(),
        "execution_method": getattr(response.attempts[-1], 'execution_metadata', {}).get('execution_method', 'unknown') if response.attempts else 'unknown'
    })
    
    if len(session.conversation_history) > 10:
        session.conversation_history = session.conversation_history[-10:]
    
    if response.success and response.final_code:
        session.last_successful_code = response.final_code
    
    session.last_prompt = request.prompt
    session.updated_at = datetime.now()

@app.get("/session/{session_id}", response_model=SessionContext)
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

@app.delete("/session/{session_id}")
async def delete_session(session_id: str, background_tasks: BackgroundTasks):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    background_tasks.add_task(sandbox.cleanup_workspace, session_id)
    del sessions[session_id]
    
    return {"message": f"Session {session_id} deleted"}

@app.get("/sessions")
async def list_sessions():
    return {
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id": session_id,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "interactions": len(session.conversation_history)
            }
            for session_id, session in sessions.items()
        ]
    }

@app.post("/validate-code")
async def validate_code_endpoint(request: dict):
    code = request.get("code", "")
    prompt = request.get("prompt", "")
    
    if not code:
        raise HTTPException(status_code=400, detail="No code provided")
    
    result = validator.validate_code(code, prompt)
    return result

@app.post("/execute-with-inputs")
async def execute_with_smart_inputs(request: dict):
    code = request.get("code", "")
    prompt = request.get("prompt", "")
    
    if not code:
        raise HTTPException(status_code=400, detail="No code provided")
    
    try:
        execution_result = validator.execute_code_with_smart_inputs(code, prompt)
        
        success_analysis = validator.validate_execution_success_only(
            code=code,
            prompt=prompt,
            execution_result=execution_result
        )
        
        return {
            "execution_result": execution_result,
            "success_analysis": success_analysis,
            "overall_success": success_analysis['overall_success']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )