import google.generativeai as genai
from typing import Optional, List, Dict
import json
import logging
import asyncio
from datetime import datetime, timedelta, date
import os
from pathlib import Path
from app.config import settings
from app.models import SessionContext
from app.utils import sanitize_code

logger = logging.getLogger(__name__)

class RateLimitError(Exception):
    pass

class RateLimiter:
    def __init__(self, daily_limit: int = 45, min_interval: float = 1.2, usage_file: str = "data/gemini_usage.json"):
        self.daily_limit = daily_limit
        self.min_interval = min_interval
        self.usage_file = Path(usage_file)
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        self.last_request_time = None
        self.usage_data = self._load_usage_data()
        
    def _load_usage_data(self) -> Dict:
        try:
            if self.usage_file.exists():
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                    if data.get('date') != str(date.today()):
                        data = self._create_fresh_usage_data()
                    return data
            else:
                return self._create_fresh_usage_data()
        except Exception as e:
            logger.warning(f"Could not load usage data: {e}")
            return self._create_fresh_usage_data()
    
    def _create_fresh_usage_data(self) -> Dict:
        return {
            'date': str(date.today()),
            'requests_made': 0,
            'last_request_time': None,
            'quota_reset_time': None
        }
    
    def _save_usage_data(self):
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save usage data: {e}")
    
    async def check_and_wait(self) -> None:
        now = datetime.now()
        
        if self.usage_data['requests_made'] >= self.daily_limit:
            remaining_time = self._time_until_reset()
            raise RateLimitError(
                f"Daily request limit ({self.daily_limit}) exceeded. "
                f"Resets in {remaining_time}"
            )
        
        if self.last_request_time:
            time_since_last = (now - self.last_request_time).total_seconds()
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        self.last_request_time = now
        
        self.usage_data['requests_made'] += 1
        self.usage_data['last_request_time'] = now.isoformat()
        self._save_usage_data()
        
        remaining = self.daily_limit - self.usage_data['requests_made']
        logger.info(f"API request made. Remaining today: {remaining}/{self.daily_limit}")
    
    def _time_until_reset(self) -> str:
        now = datetime.now()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        remaining = tomorrow - now
        
        hours, remainder = divmod(remaining.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        
        return f"{int(hours)}h {int(minutes)}m"
    
    def get_usage_stats(self) -> Dict:
        return {
            'requests_made': self.usage_data['requests_made'],
            'requests_remaining': self.daily_limit - self.usage_data['requests_made'],
            'daily_limit': self.daily_limit,
            'reset_time': self._time_until_reset(),
            'last_request': self.usage_data.get('last_request_time')
        }

class AICodeGenerator:
    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)
        
        self.rate_limiter = None
        if settings.gemini_enable_rate_limiting:
            daily_limit = settings.emergency_daily_limit if settings.emergency_mode else settings.gemini_daily_limit
            self.rate_limiter = RateLimiter(
                daily_limit=daily_limit,
                min_interval=settings.gemini_min_interval,
                usage_file=settings.usage_data_file
            )
        
        self.max_retries = settings.gemini_max_retries
        self.retry_delay = settings.gemini_retry_delay
        
        self.system_prompt = """You are an expert Python developer. Generate clean, functional Python code based on user requests.

RULES:
1. Always generate complete, runnable Python code
2. Include necessary imports at the top
3. Add error handling where appropriate
4. Use clear variable names and comments
5. For CLI output, use print() statements
6. Avoid using external libraries unless specifically requested
7. If modifying existing code, maintain the original structure when possible

The code will be executed in an isolated environment. Make sure it's safe and functional.

Respond with ONLY the Python code, no explanations or markdown formatting."""

    async def generate_code(
        self, 
        prompt: str, 
        context: Optional[SessionContext] = None,
        previous_code: Optional[str] = None,
        error_output: Optional[str] = None,
        attempt_number: int = 1
    ) -> str:
        
        try:
            if self.rate_limiter:
                await self.rate_limiter.check_and_wait()
            
            full_prompt = self._construct_prompt(
                prompt, previous_code, error_output, attempt_number, context
            )
            
            response = await self._call_gemini(full_prompt)
            generated_code = self._extract_code_from_response(response)
            
            return sanitize_code(generated_code)
            
        except RateLimitError:
            raise
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['quota', 'limit', '429']):
                if self.rate_limiter:
                    self.rate_limiter.usage_data['requests_made'] = self.rate_limiter.daily_limit
                    self.rate_limiter._save_usage_data()
                
                raise RateLimitError(
                    f"Gemini API quota exceeded. {self._get_reset_message()}"
                )
            
            raise Exception(f"AI code generation failed: {str(e)}")

    def _construct_prompt(
        self, 
        prompt: str, 
        previous_code: Optional[str], 
        error_output: Optional[str], 
        attempt_number: int,
        context: Optional[SessionContext] = None
    ) -> str:
        
        full_prompt = self.system_prompt + "\n\n"
        
        if context and context.conversation_history:
            full_prompt += "Previous conversation context:\n"
            for item in context.conversation_history[-3:]:
                if item.get("prompt"):
                    full_prompt += f"Previous request: {item['prompt']}\n"
                if item.get("code"):
                    full_prompt += f"Previous code: {item['code'][:200]}...\n"
            full_prompt += "\n"
        
        if attempt_number == 1 and not previous_code:
            full_prompt += f"Current request: {prompt}\n\nGenerate Python code:"
        
        elif previous_code and error_output:
            full_prompt += f"""The following Python code failed with an error:

{previous_code}

Error output:
{error_output}

Original request: {prompt}

Fix the code to resolve this error. Generate only the corrected Python code:"""
        
        elif previous_code:
            full_prompt += f"""Modify the following Python code:

{previous_code}

Modification request: {prompt}

Generate only the modified Python code:"""
        
        else:
            full_prompt += f"Generate Python code for: {prompt}"
        
        return full_prompt

    async def _call_gemini(self, prompt: str) -> str:
        
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                if response.text:
                    return response.text
                else:
                    raise Exception("Empty response from Gemini")
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                if any(keyword in error_msg for keyword in ['quota', 'limit', '429']):
                    logger.error(f"Gemini quota/rate limit error: {str(e)}")
                    raise
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Gemini API call failed (attempt {attempt + 1}): {str(e)}. Retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Gemini API call failed after {self.max_retries} attempts: {str(e)}")
                    raise

    def _extract_code_from_response(self, response: str) -> str:
        if not response:
            raise Exception("Empty response from AI")
        
        if self.system_prompt in response:
            response = response.replace(self.system_prompt, "").strip()
        
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        return response.strip()

    async def check_connection(self) -> bool:
        try:
            if self.rate_limiter:
                usage_stats = self.rate_limiter.get_usage_stats()
                if usage_stats['requests_remaining'] <= 0:
                    logger.warning("Cannot check connection: daily quota exhausted")
                    return False
                
                await self.rate_limiter.check_and_wait()
            
            response = self.model.generate_content("print('hello')")
            return bool(response.text)
            
        except RateLimitError:
            logger.warning("Cannot check connection: rate limit reached")
            return False
        except Exception as e:
            logger.error(f"Gemini connection check failed: {str(e)}")
            return False
    
    def get_usage_info(self) -> Dict:
        if self.rate_limiter:
            return self.rate_limiter.get_usage_stats()
        else:
            return {
                'rate_limiting_enabled': False,
                'message': 'Rate limiting is disabled in settings'
            }
    
    def reset_usage_counter(self) -> bool:
        if not self.rate_limiter:
            logger.warning("Cannot reset usage counter: rate limiting is disabled")
            return False
            
        try:
            self.rate_limiter.usage_data = self.rate_limiter._create_fresh_usage_data()
            self.rate_limiter._save_usage_data()
            logger.info("Usage counter reset successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reset usage counter: {e}")
            return False
    
    def _get_reset_message(self) -> str:
        if self.rate_limiter:
            return f"Resets in {self.rate_limiter._time_until_reset()}"
        else:
            return "Please check your Gemini API quota in the Google AI Studio console"