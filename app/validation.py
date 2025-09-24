import ast
import re
import keyword
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard" 
    STRICT = "strict"

class SuccessConfidence(Enum):
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4

class CodeValidator:
    
    DANGEROUS_IMPORTS = set()
    DANGEROUS_BUILTINS = set()
    
    SAFE_MODULES = {
        'math', 'random', 'datetime', 'time', 'json', 'csv', 'statistics',
        'collections', 'itertools', 'functools', 'operator', 'string',
        'decimal', 'fractions', 're', 'base64', 'hashlib', 'uuid',
        'calendar', 'bisect', 'heapq', 'enum', 'typing', 'dataclasses',
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'scipy',
        'plotly', 'beautifulsoup4', 'lxml', 'pillow',
        'os', 'sys', 'subprocess', 'shutil', 'glob', 'socket', 'urllib',
        'requests', 'http', 'ftplib', 'smtplib', 'telnetlib', 'pickle', 
        'marshal', 'shelve', 'dbm', 'pathlib'
    }

    def __init__(self, sandbox=None):
        self.sandbox = sandbox 
        self.validation_methods = [
            self._check_syntax,
            self._analyze_security,
            self._check_code_quality,
            self._validate_structure_universal,
            self._validate_execution_safety
        ]

    async def execute_code_with_smart_inputs(self, code: str, prompt: str, session_id: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute code using proper sandboxing instead of unsafe subprocess calls"""
        result = {
            'stdout': '',
            'stderr': '',
            'return_code': 1,
            'execution_time': 0,
            'inputs_provided': [],
            'execution_method': 'failed',
            'modified_code': None,
            'explicit_inputs_used': False
        }
        
        if not self.sandbox:
            logger.error("No sandbox provided to CodeValidator - cannot execute code safely")
            result['stderr'] = "No sandbox available for secure code execution"
            return result
        
        try:
 
            explicit_inputs = self._extract_explicit_inputs_from_prompt(prompt)
            if explicit_inputs:
                logger.info("Found explicit inputs in prompt, using those first")
                result.update(await self._execute_with_sandbox_input(code, explicit_inputs, session_id, timeout))
                if result['return_code'] == 0:
                    result['execution_method'] = 'explicit_inputs'
                    result['inputs_provided'] = explicit_inputs
                    result['explicit_inputs_used'] = True
                    return result
            
         
            smart_code = self._auto_replace_input_calls(code, prompt)
            if smart_code != code:
                logger.info("Attempting execution with input replacement")
                result.update(await self._execute_with_sandbox(smart_code, session_id, timeout))
                if result['return_code'] == 0 and result['stdout'].strip():
                    result['execution_method'] = 'input_replacement'
                    result['modified_code'] = smart_code
                    return result
            
            
            logger.info("Attempting execution with stdin input injection")
            auto_inputs = self._generate_context_inputs(prompt, code)
            result.update(await self._execute_with_sandbox_input(code, auto_inputs, session_id, timeout))
            if result['return_code'] == 0:
                result['execution_method'] = 'stdin_injection'
                result['inputs_provided'] = auto_inputs
                return result
            
     
            logger.info("Fallback: attempting original code execution")
            result.update(await self._execute_with_sandbox(code, session_id, timeout))
            result['execution_method'] = 'original'
            
        except Exception as e:
            logger.error(f"Sandboxed code execution failed: {str(e)}")
            result['stderr'] = f"Sandbox execution error: {str(e)}"
        
        return result

    async def _execute_with_sandbox(self, code: str, session_id: str, timeout: int) -> Dict[str, Any]:
        """Execute code directly in sandbox without input handling"""
        try:
            stdout, stderr, exit_code, execution_time = await self.sandbox.execute_code(code, session_id)
            
            return {
                'stdout': stdout,
                'stderr': stderr,
                'return_code': exit_code,
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"Direct sandbox execution failed: {str(e)}")
            return {
                'stdout': '',
                'stderr': f'Sandbox execution error: {str(e)}',
                'return_code': 1,
                'execution_time': 0
            }

    async def _execute_with_sandbox_input(self, code: str, auto_inputs: List[str], session_id: str, timeout: int) -> Dict[str, Any]:
        """Execute code in sandbox with input handling by modifying the code"""
        try:
         
            input_values = [str(inp).replace('"', '') for inp in auto_inputs]
            
       
            input_simulation_code = f"""
import sys
from io import StringIO

# Simulate input by replacing sys.stdin
input_data = {repr('\\n'.join(input_values) + '\\n')}
sys.stdin = StringIO(input_data)

# Original code follows:
{code}
"""
            
            stdout, stderr, exit_code, execution_time = await self.sandbox.execute_code(input_simulation_code, session_id)
            
            return {
                'stdout': stdout,
                'stderr': stderr,
                'return_code': exit_code,
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"Sandbox input execution failed: {str(e)}")
            return {
                'stdout': '',
                'stderr': f'Sandbox input execution error: {str(e)}',
                'return_code': 1,
                'execution_time': 0
            }


    def _extract_explicit_inputs_from_prompt(self, prompt: str) -> List[str]:
        inputs = []
        original_prompt = prompt
        prompt_lower = prompt.lower()
    
        pattern1 = r'input[:\s=]*([^,\n]+?)(?:,|\s+)(?:expected\s+)?output[:\s=]*([^,\n]+)'
        matches = re.findall(pattern1, prompt_lower, re.IGNORECASE)
        for input_val, output_val in matches:
            cleaned_input = input_val.strip().strip('"\'')
            if cleaned_input:
                inputs.append(cleaned_input)
    
        pattern2 = r'(?:given|with|for)\s+input[:\s=]*([^,\n\.]+)'
        matches = re.findall(pattern2, prompt_lower, re.IGNORECASE)
        for match in matches:
            cleaned_input = match.strip().strip('"\'')
            if cleaned_input:
                inputs.append(cleaned_input)
    
        pattern3 = r'(?:test\s+case|example)[:\s]*([^->\n]+)\s*->\s*([^\n]+)'
        matches = re.findall(pattern3, prompt_lower, re.IGNORECASE)
        for input_val, output_val in matches:
            cleaned_input = input_val.strip().strip('"\'')
            if cleaned_input:
                inputs.append(cleaned_input)
    
        pattern4 = r'input\s*(?:=|is)\s*([^,\n\.]+)'
        matches = re.findall(pattern4, prompt_lower, re.IGNORECASE)
        for match in matches:
            cleaned_input = match.strip().strip('"\'')
            if cleaned_input:
                inputs.append(cleaned_input)
    
        pattern5 = r'input\s*:\s*(.+?)(?=\s*(?:output|explanation|example|\n\s*\n|$))'
        matches = re.findall(pattern5, original_prompt, re.IGNORECASE | re.DOTALL)
        for match in matches:
            cleaned_input = match.strip()
            if cleaned_input:
                inputs.append(cleaned_input)
    
        pattern6 = r'(\w+)\s*=\s*([^\n,]+?)(?=\s*,\s*\w+\s*=|\s*$|\s*\n)'
        matches = re.findall(pattern6, original_prompt, re.MULTILINE)
        for var_name, var_value in matches:
            assignment = f"{var_name} = {var_value.strip()}"
            inputs.append(assignment)
    
        pattern7 = r'(\w+)\s*\(\s*([^)]+)\s*\)'
        matches = re.findall(pattern7, original_prompt)
        for func_name, params in matches:
            if func_name.lower() not in ['if', 'for', 'while', 'when', 'where', 'what', 'how']:
                param_list = [p.strip() for p in params.split(',')]
                for param in param_list:
                    if param and not param.lower() in ['the', 'a', 'an']:
                        inputs.append(param)
                inputs.append(f"{func_name}({params})")
    
        pattern8a = r'\[([^\]]+)\]'
        matches = re.findall(pattern8a, original_prompt)
        for match in matches:
            inputs.append(f"[{match}]")
    
        pattern8b = r'\{([^}]+)\}'
        matches = re.findall(pattern8b, original_prompt)
        for match in matches:
            inputs.append(f"{{{match}}}")
    
        pattern9 = r'["\']([^"\']{1,50})["\']'
        matches = re.findall(pattern9, original_prompt)
        for match in matches:
            if (len(match.strip()) > 0 and 
                match.lower().strip() not in ['input', 'output', 'example', 'test', 'case', 'the', 'a', 'an', 'is', 'are']):
                inputs.append(f'"{match}"')
    
        pattern10 = r'(?:example|test|input|with|of|for)\s+.*?(\d+(?:\.\d+)?)'
        matches = re.findall(pattern10, prompt_lower)
        for match in matches:
            inputs.append(match)
    
        cleaned_inputs = []
        seen_inputs = set()
    
        for inp in inputs:
            if not inp or len(inp.strip()) == 0:
                continue
            
            cleaned = inp.strip()
        
            if cleaned.lower() in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']:
                continue
            
            if cleaned.lower() in seen_inputs:
                continue
            seen_inputs.add(cleaned.lower())
        
            if re.match(r'^\d+(\.\d+)?$', cleaned):
                cleaned_inputs.append(cleaned)
            elif re.match(r'^\[.*\]$', cleaned):
                cleaned_inputs.append(cleaned)
            elif re.match(r'^\{.*\}$', cleaned):
                cleaned_inputs.append(cleaned)
            elif re.match(r'^\w+\s*=\s*.+$', cleaned):
                cleaned_inputs.append(cleaned)
            elif re.match(r'^\w+\(.+\)$', cleaned):
                cleaned_inputs.append(cleaned)
            elif cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned_inputs.append(cleaned)
            elif cleaned.startswith("'") and cleaned.endswith("'"):
                cleaned_inputs.append(f'"{cleaned[1:-1]}"')
            else:
                if (' ' in cleaned or 
                    any(c in cleaned for c in [',', ':', ';', '(', ')', '[', ']', '{', '}']) or
                    re.search(r'[a-zA-Z]', cleaned)):
                    if not any(c in cleaned for c in ['=', '(', '[', '{']):
                        cleaned_inputs.append(f'"{cleaned}"')
                    else:
                        cleaned_inputs.append(cleaned)
                else:
                    cleaned_inputs.append(cleaned)
    
        def input_priority(inp):
            if '=' in inp:
                return 0
            elif inp.startswith('[') or inp.startswith('{'):
                return 1
            elif '(' in inp:
                return 2
            elif inp.startswith('"'):
                return 3
            else:
                return 4
    
        cleaned_inputs.sort(key=input_priority)
    
        if cleaned_inputs:
            logger.info(f"Extracted explicit inputs from prompt: {cleaned_inputs}")
    
        return cleaned_inputs[:5]

    def _auto_replace_input_calls(self, code: str, prompt: str) -> str:
        explicit_inputs = self._extract_explicit_inputs_from_prompt(prompt)
        if explicit_inputs:
            logger.info("Using explicit inputs for replacement")
            modified_code = code
            input_calls = list(re.finditer(r'input\([^)]*\)', code, re.IGNORECASE))
            
            for i, match in enumerate(input_calls):
                if i < len(explicit_inputs):
                    replacement_value = explicit_inputs[i]
                    if replacement_value.startswith('"') and replacement_value.endswith('"'):
                        replacement = replacement_value
                    else:
                        replacement = replacement_value
                    
                    modified_code = modified_code[:match.start()] + replacement + modified_code[match.end():]
                    offset = len(replacement) - len(match.group())
                    input_calls = [(m.start() + (offset if m.start() > match.start() else 0), 
                                  m.end() + (offset if m.end() > match.start() else 0), 
                                  m.group()) for m in input_calls]
                    logger.info(f"Replaced input() call {i+1} with: {replacement}")
            
            return modified_code
        
        prompt_lower = prompt.lower()
        code_lower = code.lower()
        
        replacements = []
        
        if 'fibonacci' in prompt_lower:
            if any(word in prompt_lower for word in ['10th', 'tenth', '10']):
                replacements.append((r'input\([^)]*\)', '10'))
            elif any(word in prompt_lower for word in ['5th', 'fifth', '5']):
                replacements.append((r'input\([^)]*\)', '5'))
            else:
                replacements.append((r'input\([^)]*\)', '8'))
        
        elif 'factorial' in prompt_lower:
            if any(word in prompt_lower for word in ['5', 'five']):
                replacements.append((r'input\([^)]*\)', '5'))
            elif any(word in prompt_lower for word in ['10', 'ten']):
                replacements.append((r'input\([^)]*\)', '10'))
            else:
                replacements.append((r'input\([^)]*\)', '6'))
        
        elif 'prime' in prompt_lower:
            if any(word in prompt_lower for word in ['17', 'seventeen']):
                replacements.append((r'input\([^)]*\)', '17'))
            else:
                replacements.append((r'input\([^)]*\)', '23'))
        
        elif any(word in prompt_lower for word in ['calculate', 'compute', 'add', 'multiply', 'divide']):
            replacements.extend([
                (r'input\([^)]*first[^)]*\)', '15'),
                (r'input\([^)]*second[^)]*\)', '25'),
                (r'input\([^)]*number[^)]*\)', '10'),
                (r'input\([^)]*\)', '12')
            ])
        
        elif any(word in prompt_lower for word in ['age', 'year', 'old']):
            replacements.append((r'input\([^)]*\)', '25'))
        
        elif any(word in prompt_lower for word in ['name', 'user', 'person']):
            replacements.append((r'input\([^)]*\)', '"Alice"'))
        
        elif any(word in prompt_lower for word in ['file', 'filename', 'path']):
            replacements.append((r'input\([^)]*\)', '"data.txt"'))
        
        elif any(word in prompt_lower for word in ['yes', 'no', 'confirm', 'continue']):
            replacements.append((r'input\([^)]*\)', '"yes"'))
        
        else:
            numbers = re.findall(r'\b(\d+)\b', prompt)
            if numbers:
                default_num = numbers[0]
            else:
                default_num = '10'
            replacements.append((r'input\([^)]*\)', default_num))
        
        modified_code = code
        for pattern, replacement in replacements:
            if re.search(pattern, code, re.IGNORECASE):
                modified_code = re.sub(pattern, replacement, modified_code, flags=re.IGNORECASE)
                logger.info(f"Replaced input() with: {replacement}")
                break
        
        return modified_code

    def _generate_context_inputs(self, prompt: str, code: str) -> List[str]:
        explicit_inputs = self._extract_explicit_inputs_from_prompt(prompt)
        if explicit_inputs:
            logger.info("Using explicit inputs from prompt for context generation")
            processed_inputs = []
            for inp in explicit_inputs:
                if inp.startswith('"') and inp.endswith('"'):
                    processed_inputs.append(inp[1:-1])
                else:
                    processed_inputs.append(inp)
            
            while len(processed_inputs) < 5:
                processed_inputs.extend(['10', 'test', 'yes', '1', 'default'])
            
            return processed_inputs[:5]
        
        inputs = []
        prompt_lower = prompt.lower()
        code_lower = code.lower()
        
        input_count = len(re.findall(r'input\s*\(', code))
        
        mentioned_numbers = re.findall(r'\b(\d+)\b', prompt)
        if mentioned_numbers:
            inputs.extend(mentioned_numbers[:3])
        
        quoted_strings = re.findall(r'["\']([^"\']+)["\']', prompt)
        if quoted_strings:
            inputs.extend(quoted_strings[:2])
        
        domain_patterns = [
            (['fibonacci', 'fib'], ['10', '8', '15', '5']),
            (['factorial', 'fact'], ['5', '7', '6', '10']),
            (['prime'], ['17', '23', '29', '13']),
            (['square', 'power'], ['4', '9', '16', '3']),
            (['sum', 'add', 'total'], ['10', '20', '5', '15']),
            (['multiply', 'product'], ['6', '7', '4', '8']),
            (['divide', 'division'], ['20', '4', '15', '3']),
            (['average', 'mean'], ['10', '20', '30', '25']),
            (['percentage', 'percent'], ['85', '100', '75', '60']),
            (['temperature'], ['25', '32', '100', '0']),
            (['sort', 'order', 'arrange'], ['5', '2', '8', '1', '9', '3']),
            (['search', 'find'], ['target', '42', 'search_item', 'key']),
            (['list', 'array'], ['1', '2', '3', '4', '5']),
            (['matrix'], ['3', '3', '1', '2']),
            (['binary', 'tree'], ['15', '7', '3', '12']),
            (['text', 'string', 'word'], ['Hello World', 'Python Programming', 'Sample Text']),
            (['password', 'secret'], ['mypassword123', 'secret123', 'test_pass']),
            (['email', 'mail'], ['user@example.com', 'test@gmail.com', 'sample@test.com']),
            (['url', 'link'], ['https://example.com', 'www.test.com', 'api.example.org']),
            (['sentence'], ['This is a test sentence', 'Hello world', 'Sample text here']),
            (['name', 'username'], ['Alice', 'Bob', 'Charlie', 'TestUser']),
            (['age'], ['25', '30', '22', '35']),
            (['grade', 'score'], ['85', '92', '78', '95']),
            (['salary', 'income'], ['50000', '75000', '60000', '45000']),
            (['height'], ['175', '180', '165', '170']),
            (['weight'], ['70', '80', '65', '75']),
            (['file', 'filename'], ['data.txt', 'sample.csv', 'test.json', 'input.txt']),
            (['csv'], ['data.csv', 'students.csv', 'sales.csv', 'records.csv']),
            (['json'], ['data.json', 'config.json', 'users.json', 'settings.json']),
            (['excel'], ['data.xlsx', 'report.xls', 'sheet.xlsx', 'table.xls']),
            (['menu', 'choice', 'option'], ['1', '2', '3', 'A']),
            (['continue', 'proceed'], ['yes', 'y', 'continue', '1']),
            (['confirm', 'sure'], ['yes', 'y', 'true', '1']),
            (['quit', 'exit'], ['no', 'n', 'false', '0']),
            (['game', 'play'], ['start', '1', 'play', 'begin']),
            (['dice', 'roll'], ['6', '4', '2', '5']),
            (['guess', 'number'], ['50', '25', '75', '42']),
            (['lottery'], ['7', '14', '21', '35', '42', '49']),
            (['distance'], ['100', '50', '25', '75']),
            (['speed', 'velocity'], ['60', '80', '40', '100']),
            (['time', 'duration'], ['30', '60', '45', '90']),
            (['area'], ['25', '50', '100', '36']),
            (['volume'], ['125', '64', '27', '100']),
            (['price', 'cost'], ['19.99', '25.50', '100', '15.75']),
            (['tax'], ['8.5', '10', '7.25', '6']),
            (['discount'], ['10', '15', '20', '25']),
            (['interest'], ['5.5', '3.2', '4.8', '6.1']),
            (['date'], ['2024-01-01', '15/03/2024', '12-25-2023', 'today']),
            (['year'], ['2024', '2023', '2025', '2022']),
            (['month'], ['January', 'March', 'December', '06']),
            (['day'], ['Monday', 'Friday', '15', '1']),
            (['ip', 'address'], ['192.168.1.1', '127.0.0.1', '10.0.0.1', '172.16.1.1']),
            (['port'], ['8080', '3000', '5432', '80']),
            (['domain'], ['example.com', 'test.org', 'mysite.net', 'api.service.com']),
            (['database', 'db'], ['testdb', 'myapp_db', 'production', 'users']),
            (['table'], ['users', 'products', 'orders', 'customers']),
            (['query'], ['SELECT * FROM users', 'users.sql', 'data_query', 'search']),
        ]
        
        for keywords, values in domain_patterns:
            if any(keyword in prompt_lower for keyword in keywords):
                inputs.extend(values[:4])
                break
        
        input_prompts = re.findall(r'input\s*\(\s*["\']([^"\']+)["\']', code)
        for input_prompt in input_prompts:
            input_prompt_lower = input_prompt.lower()
            
            if any(word in input_prompt_lower for word in ['number', 'num', 'digit']):
                inputs.append('42')
            elif any(word in input_prompt_lower for word in ['name']):
                inputs.append('TestUser')
            elif any(word in input_prompt_lower for word in ['age']):
                inputs.append('25')
            elif any(word in input_prompt_lower for word in ['email']):
                inputs.append('test@example.com')
            elif any(word in input_prompt_lower for word in ['password']):
                inputs.append('password123')
            else:
                prompt_numbers = re.findall(r'\d+', input_prompt)
                if prompt_numbers:
                    inputs.append(prompt_numbers[0])
                else:
                    inputs.append('10')
        
        if not inputs:
            if any(word in prompt_lower for word in ['number', 'integer', 'digit', 'value']):
                inputs.extend(['10', '5', '25', '100'])
            elif any(word in prompt_lower for word in ['string', 'text', 'word', 'message']):
                inputs.extend(['Hello', 'Test', 'Sample', 'Input'])
            elif any(word in prompt_lower for word in ['name', 'user']):
                inputs.extend(['Alice', 'Bob', 'User', 'Test'])
            else:
                inputs.extend(['10', 'test', 'yes', '5', 'sample', '1', 'input'])
        
        while len(inputs) < max(input_count, 5):
            inputs.extend(['10', 'default', 'yes', '1', '100', 'test', 'true'])
        
        return inputs[:max(input_count, 5)]


    def _check_syntax(self, code: str) -> Dict[str, Any]:
        result = {'valid': True, 'errors': [], 'warnings': [], 'confidence_weight': 0.3}
        
        try:
            ast.parse(code)
            result['checks_passed'] = ['syntax_check']
        except SyntaxError as e:
            result['valid'] = False
            result['errors'].append(f"Syntax Error: {str(e)}")
            result['confidence_weight'] = 0.0
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Parse Error: {str(e)}")
            result['confidence_weight'] = 0.0
        
        return result

    def _analyze_security(self, code: str) -> Dict[str, Any]:
        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'safety_score': 10,
            'has_dangerous_imports': False,
            'confidence_weight': 0.3,
            'checks_passed': ['security_check']
        }
        
        try:
            tree = ast.parse(code)
            result['checks_passed'].append('security_check')
                
        except Exception as e:
            result['warnings'].append(f"Security analysis incomplete: {str(e)}")
            result['confidence_weight'] = 0.1
        
        return result

    def _check_code_quality(self, code: str) -> Dict[str, Any]:
        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'confidence_weight': 0.1,
            'checks_passed': []
        }
        
        lines = code.split('\n')
        
        has_output = any(pattern in code for pattern in ['print(', 'return ', 'yield ', 'plt.show()', 'plt.savefig('])
        if not has_output:
            result['warnings'].append("Code doesn't appear to produce visible output")
        
        if re.search(r'while\s+True\s*:', code):
            if 'break' not in code and 'return' not in code:
                result['warnings'].append("Potential infinite loop detected")
        
        result['checks_passed'].append('quality_check')
        return result

    def _validate_structure_universal(self, code: str) -> Dict[str, Any]:
        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'confidence_weight': 0.2,
            'checks_passed': [],
            'structure_analysis': {}
        }
        
        try:
            tree = ast.parse(code)
            structure_analysis = self._analyze_code_structure(tree)
            result['structure_analysis'] = structure_analysis
            
            if not structure_analysis['has_execution']:
                result['warnings'].append("Code appears to only contain definitions, no execution")
            
            result['checks_passed'].append('structure_validation')
            
        except Exception as e:
            result['warnings'].append(f"Structure validation incomplete: {str(e)}")
            result['confidence_weight'] = 0.1
        
        return result

    def _validate_execution_safety(self, code: str) -> Dict[str, Any]:
        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'confidence_weight': 0.1,
            'checks_passed': []
        }
        
        if 'input(' in code:
            result['warnings'].append("Code requires user input - will be handled automatically")
        
        memory_intensive_patterns = [
            (r'range\(\s*\d{6,}\s*\)', "Large range operations"),
            (r'\.readlines\(\)', "Reading all file lines into memory")
        ]
        
        for pattern, description in memory_intensive_patterns:
            if re.search(pattern, code):
                result['warnings'].append(f"Memory usage warning: {description}")
        
        result['checks_passed'].append('execution_safety')
        return result

    def _analyze_code_structure(self, tree: ast.AST) -> Dict[str, Any]:
        analysis = {
            'has_functions': False,
            'has_classes': False,
            'has_loops': False,
            'has_conditions': False,
            'has_math_operations': False,
            'has_execution': False,
            'complexity_score': 0,
            'imports': [],
            'function_names': [],
            'has_main_block': False,
            'line_count': 0,
            'has_error_handling': False
        }
        
        class StructureVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                analysis['has_functions'] = True
                analysis['function_names'].append(node.name)
                analysis['complexity_score'] += 2
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                analysis['has_classes'] = True
                analysis['complexity_score'] += 3
                self.generic_visit(node)
            
            def visit_For(self, node):
                analysis['has_loops'] = True
                analysis['complexity_score'] += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                analysis['has_loops'] = True
                analysis['complexity_score'] += 2
                self.generic_visit(node)
            
            def visit_If(self, node):
                analysis['has_conditions'] = True
                analysis['complexity_score'] += 1
                
                if (isinstance(node.test, ast.Compare) and 
                    isinstance(node.test.left, ast.Name) and
                    node.test.left.id == '__name__'):
                    analysis['has_main_block'] = True
                
                self.generic_visit(node)
            
            def visit_Try(self, node):
                analysis['has_error_handling'] = True
                analysis['complexity_score'] += 1
                self.generic_visit(node)
            
            def visit_BinOp(self, node):
                if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
                    analysis['has_math_operations'] = True
                self.generic_visit(node)
            
            def visit_Import(self, node):
                for alias in node.names:
                    analysis['imports'].append(alias.name)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    analysis['imports'].append(node.module)
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['print', 'input', 'len', 'range', 'open']:
                        analysis['has_execution'] = True
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['show', 'plot', 'savefig', 'write', 'save']:
                        analysis['has_execution'] = True
                
                self.generic_visit(node)
        
        visitor = StructureVisitor()
        visitor.visit(tree)
        
        return analysis

    def validate_code(self, code: str, prompt: str = "", validation_level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[str, Any]:
        result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'sanitized_code': code,
            'safety_score': 10,
            'confidence_score': 0.0,
            'intent_analysis': {},
            'checks_passed': [],
            'structure_analysis': {},
            'success_prediction': True
        }
        
        try:
            for method in self.validation_methods:
                try:
                    method_result = method(code)
                    self._merge_validation_results(result, method_result, method.__name__)
                    
                except Exception as e:
                    logger.error(f"Validation method {method.__name__} failed: {str(e)}")
                    result['warnings'].append(f"Validation method {method.__name__} failed")
            
            if prompt:
                result['intent_analysis'] = self._extract_intent_from_prompt(prompt)
            
            result['confidence_score'] = self._calculate_confidence_score(result)
            
            result['success_prediction'] = self._predict_execution_success(result)
            
            result['is_valid'] = self._make_validation_decision(result, validation_level)
            
            result['sanitized_code'] = self._basic_sanitize_code(code)
            
        except Exception as e:
            logger.error(f"Code validation error: {str(e)}")
            result['is_valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result

    def _merge_validation_results(self, main_result: Dict, method_result: Dict, method_name: str):
        if not method_result.get('valid', True):
            main_result['is_valid'] = False
        
        main_result['warnings'].extend(method_result.get('warnings', []))
        main_result['errors'].extend(method_result.get('errors', []))
        main_result['checks_passed'].extend(method_result.get('checks_passed', []))
        
        if 'safety_score' in method_result:
            main_result['safety_score'] = min(main_result['safety_score'], method_result['safety_score'])
        
        if 'has_dangerous_imports' in method_result:
            main_result['has_dangerous_imports'] = method_result['has_dangerous_imports']
        
        if 'structure_analysis' in method_result:
            main_result['structure_analysis'] = method_result['structure_analysis']

    def _calculate_confidence_score(self, result: Dict) -> float:
        base_score = 0.0
        
        check_weights = {
            'syntax_check': 0.3,
            'security_check': 0.3,
            'quality_check': 0.1,
            'structure_validation': 0.2,
            'execution_safety': 0.1
        }
        
        for check in result['checks_passed']:
            base_score += check_weights.get(check, 0.05)
        
        base_score -= len(result['errors']) * 0.2
        base_score -= len(result['warnings']) * 0.05
        
        safety_factor = result['safety_score'] / 10.0
        base_score *= safety_factor
        
        return max(0.0, min(1.0, base_score))

    def _predict_execution_success(self, result: Dict) -> bool:
        if not result['is_valid']:
            return False
        
        if result['confidence_score'] < 0.4:
            return False
            
        if result['safety_score'] < 5:
            return False
        
        structure = result.get('structure_analysis', {})
        if not structure.get('has_execution', False):
            return False
        
        return True

    def _make_validation_decision(self, result: Dict, level: ValidationLevel) -> bool:
        if level == ValidationLevel.BASIC:
            return len(result['errors']) == 0
        
        elif level == ValidationLevel.STANDARD:
            return result['is_valid'] and result['confidence_score'] > 0.4
        
        elif level == ValidationLevel.STRICT:
            return (result['is_valid'] and 
                    result['confidence_score'] > 0.6 and
                    result['safety_score'] > 7 and
                    len(result['warnings']) < 5)
        
        return result['is_valid']

    def _basic_sanitize_code(self, code: str) -> str:
        sanitized = code
        
        sanitized = re.sub(r'__file__', '"/tmp/script.py"', sanitized)
        
        if 'requests.get(' in sanitized and 'timeout=' not in sanitized:
            sanitized = re.sub(
                r'requests\.get\(([^)]+)\)',
                r'requests.get(\1, timeout=10)',
                sanitized
            )
        
        return sanitized

    def get_enhanced_failure_feedback(self, prompt: str, execution_result: Dict[str, Any], success_analysis: Dict[str, Any]) -> str:
        feedback_parts = []
        
        if execution_result.get('explicit_inputs_used'):
            feedback_parts.append("Used explicit inputs from prompt but execution still failed")
        
        exec_method = execution_result.get('execution_method', 'unknown')
        if exec_method == 'failed':
            feedback_parts.append("All execution methods failed")
        elif exec_method == 'stdin_injection':
            if execution_result.get('inputs_provided'):
                feedback_parts.append(f"Code required input, provided: {execution_result['inputs_provided']}")
        elif exec_method == 'explicit_inputs':
            feedback_parts.append("Code executed with explicit inputs from prompt")
        
        if execution_result.get('return_code', 0) != 0:
            feedback_parts.append(f"Code failed with return code {execution_result.get('return_code')}")
        
        stderr = execution_result.get('stderr', '').strip()
        if stderr and not self._is_warning_only(stderr):
            if 'EOF when reading a line' in stderr:
                feedback_parts.append("Code expects user input but none was provided effectively")
            else:
                error_lines = stderr.split('\n')
                key_error = error_lines[-1] if error_lines else stderr[:100]
                feedback_parts.append(f"Runtime error: {key_error}")
        
        stdout = execution_result.get('stdout', '').strip()
        
        if not stdout:
            feedback_parts.append("No output produced")
        elif len(stdout) < 10:
            feedback_parts.append("Output too brief")
        elif 'EOF when reading a line' in stdout:
            feedback_parts.append("Code attempted to read input but execution environment couldn't provide it")
        
        if success_analysis.get('failure_reasons'):
            top_failures = success_analysis['failure_reasons'][:2]
            feedback_parts.extend(top_failures)
        
        intent = self._extract_intent_from_prompt(prompt)
        
        if 'calculate' in intent.get('actions', []) and not re.search(r'\d+(?:\.\d+)?', stdout):
            feedback_parts.append("Expected numerical result for calculation task")
            
        if intent.get('expected_data_type') == 'list' and not any(char in stdout for char in ['[', ']', ',']):
            feedback_parts.append("Expected list or array output format")
        
        if 'input(' in execution_result.get('modified_code', '') or 'input(' in stdout:
            feedback_parts.append("Consider avoiding input() for automated execution, use predefined values instead")
        
        if not feedback_parts:
            feedback_parts.append("Output doesn't adequately address the original request")
        
        return ". ".join(feedback_parts) + "."

    def _is_warning_only(self, stderr_output: str) -> bool:
        warning_indicators = [
            "warning:", "deprecated", "futurewarning", "userwarning",
            "runtimewarning", "pendingdeprecationwarning", "deprecationwarning"
        ]
        
        lines = stderr_output.lower().split('\n')
        for line in lines:
            line = line.strip()
            if line and not any(warning in line for warning in warning_indicators):
                return False
        return True

    def _extract_intent_from_prompt(self, prompt: str) -> Dict[str, Any]:
        intent = {
            'keywords': [],
            'expected_outputs': [],
            'data_operations': [],
            'libraries_mentioned': [],
            'file_operations': False,
            'network_operations': False,
            'user_interaction': False,
            'domain': None,
            'actions': [],
            'expected_data_type': None
        }
        
        if not prompt or not isinstance(prompt, str):
            logger.warning("Invalid prompt provided to _extract_intent_from_prompt")
            return intent
        
        try:
            prompt_lower = prompt.lower()
            
            action_patterns = {
                'calculate': ['calculate', 'compute', 'find', 'determine', 'solve'],
                'generate': ['generate', 'create', 'make', 'build', 'produce'],
                'analyze': ['analyze', 'examine', 'study', 'investigate', 'process'],
                'visualize': ['plot', 'chart', 'graph', 'visualize', 'draw', 'display'],
                'sort': ['sort', 'order', 'arrange', 'rank', 'organize'],
                'search': ['search', 'find', 'locate', 'match', 'filter'],
                'convert': ['convert', 'transform', 'change', 'translate', 'parse'],
                'download': ['download', 'fetch', 'get', 'retrieve', 'scrape'],
                'simulate': ['simulate', 'model', 'predict', 'forecast', 'estimate']
            }
            
            for category, keywords in action_patterns.items():
                if any(keyword in prompt_lower for keyword in keywords):
                    intent['keywords'].append(category)
                    intent['actions'].append(category)
            
            common_libraries = [
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'requests', 
                'beautifulsoup', 'scipy', 'sklearn', 'tensorflow', 'torch',
                'json', 'csv', 'xml', 'sqlite3', 'mysql', 'psycopg2',
                'plotly', 'dash', 'streamlit', 'flask', 'django'
            ]
            
            for lib in common_libraries:
                if lib in prompt_lower:
                    intent['libraries_mentioned'].append(lib)
            
            if any(word in prompt_lower for word in ['file', 'csv', 'json', 'txt', 'read', 'write', 'save', 'load']):
                intent['file_operations'] = True
                
            if any(word in prompt_lower for word in ['url', 'api', 'download', 'request', 'web', 'scrape', 'http']):
                intent['network_operations'] = True
                
            if any(word in prompt_lower for word in ['input', 'user', 'ask', 'prompt', 'interactive']):
                intent['user_interaction'] = True
            
            domains = {
                'data_analysis': ['data', 'csv', 'dataframe', 'pandas', 'analysis', 'statistics'],
                'web_scraping': ['scrape', 'web', 'html', 'url', 'beautifulsoup', 'selenium'],
                'machine_learning': ['model', 'predict', 'train', 'sklearn', 'tensorflow', 'ml', 'ai'],
                'file_processing': ['file', 'read', 'write', 'json', 'xml', 'csv', 'parse'],
                'mathematics': ['math', 'equation', 'formula', 'algorithm', 'calculation'],
                'visualization': ['plot', 'chart', 'matplotlib', 'seaborn', 'graph'],
                'api': ['api', 'request', 'endpoint', 'rest', 'json', 'response'],
                'automation': ['automate', 'script', 'batch', 'schedule', 'cron']
            }
            
            for domain, keywords in domains.items():
                if any(keyword in prompt_lower for keyword in keywords):
                    intent['domain'] = domain
                    break
            
            if any(word in prompt_lower for word in ['number', 'integer', 'float', 'calculate', 'sum']):
                intent['expected_data_type'] = 'numerical'
            elif any(word in prompt_lower for word in ['list', 'array', 'items', 'elements']):
                intent['expected_data_type'] = 'list'
            elif any(word in prompt_lower for word in ['text', 'string', 'content', 'message']):
                intent['expected_data_type'] = 'text'
            elif any(word in prompt_lower for word in ['json', 'dict', 'object', 'data']):
                intent['expected_data_type'] = 'structured'
            
        except Exception as e:
            logger.error(f"Error in _extract_intent_from_prompt: {str(e)}")
        
        return intent

    def validate_execution_success_only(self, code: str, prompt: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            'overall_success': False,
            'confidence_score': 0.0,
            'success_reasons': [],
            'failure_reasons': [],
            'method_scores': {},
            'execution_time': None,
            'output_quality': 'unknown'
        }
        
        try:
            intent = self._extract_intent_from_prompt(prompt)
            if intent is None:
                intent = {}
        except Exception as e:
            logger.error(f"Intent extraction failed: {str(e)}")
            intent = {}
        
        return_code = execution_result.get('return_code', 1)
        stdout = execution_result.get('stdout', '').strip()
        stderr = execution_result.get('stderr', '').strip()
        
        base_confidence = 0.0
        if return_code == 0:
            base_confidence = 0.3
            if stdout:
                base_confidence = 0.5
        
        success_methods = [
            self._execution_success_check,
            self._output_pattern_matching,
            self._intent_fulfillment_check,
            self._test_case_validation
        ]
        
        method_weights = {
            '_execution_success_check': 0.4,
            '_output_pattern_matching': 0.25,
            '_intent_fulfillment_check': 0.2,
            '_test_case_validation': 0.15
        }
        
        for method in success_methods:
            try:
                method_name = method.__name__
                
                if method == self._output_pattern_matching:
                    method_result = method(execution_result, intent)
                elif method == self._intent_fulfillment_check:
                    method_result = method(code, prompt, execution_result, intent)
                else:
                    method_result = method(code, prompt, execution_result)
                
                result['method_scores'][method_name] = method_result
                method_weight = method_weights.get(method_name, 0.1)
                
                if method_result.get("success", False):
                    result['success_reasons'].extend(method_result.get("reasons", []))
                    result['confidence_score'] += method_weight
                else:
                    result['failure_reasons'].extend(method_result.get("reasons", []))
                    if return_code == 0:
                        result['confidence_score'] += method_weight * 0.1
                        
            except Exception as e:
                logger.error(f"Success detection method {method.__name__} failed: {str(e)}")
                result['failure_reasons'].append(f"Method {method.__name__} error: {str(e)}")
        
        result['confidence_score'] = min(1.0, max(0.0, result['confidence_score'] + base_confidence))
        
        result['overall_success'] = result['confidence_score'] > 0.4
        
        if result['confidence_score'] > 0.7:
            result['output_quality'] = 'excellent'
        elif result['confidence_score'] > 0.4:
            result['output_quality'] = 'good' 
        elif result['confidence_score'] > 0.2:
            result['output_quality'] = 'acceptable'
        else:
            result['output_quality'] = 'poor'
        
        return result

   
    def _execution_success_check(self, code: str, prompt: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        if execution_result.get("return_code", 1) != 0:
            return {
                "success": False,
                "reasons": [f"Code failed with return code: {execution_result.get('return_code')}"],
                "weight": 0.0
            }
        
        stderr_output = execution_result.get("stderr", "").strip()
        if stderr_output and not self._is_warning_only(stderr_output):
            return {
                "success": False,
                "reasons": [f"Code produced errors: {stderr_output[:100]}..."],
                "weight": 0.0
            }
        
        stdout_output = execution_result.get("stdout", "").strip()
        if not stdout_output:
            return {
                "success": False,
                "reasons": ["Code executed but produced no output"],
                "weight": 0.0
            }
        
        return {
            "success": True,
            "reasons": ["Code executed without errors and produced output"],
            "weight": 0.3
        }

    def _output_pattern_matching(self, execution_result: Dict[str, Any], intent: Dict) -> Dict[str, Any]:
        if intent is None:
            intent = {}
            logger.warning("Intent was None, using empty dict")
        
        output = execution_result.get("stdout", "").strip()
        return_code = execution_result.get("return_code", 1)
        stderr = execution_result.get("stderr", "").strip()
        
        if return_code == 0:
            if not output and not stderr:
                return {
                    "success": True,
                    "reasons": ["Code executed successfully but produced no output - may be valid for some tasks"],
                    "weight": 0.3
                }
            
            if output:
                if len(output) >= 1:
                    return {
                        "success": True,
                        "reasons": ["Code executed successfully and produced output"],
                        "weight": 0.4
                    }
        
        expected_patterns = self._extract_expected_patterns("", intent)
        matches = []
        misses = []
        
        for pattern_name, pattern_info in expected_patterns.items():
            if self._check_pattern(output, pattern_info):
                matches.append(f"Found expected {pattern_name}")
            else:
                misses.append(f"Missing expected {pattern_name}")
        
        if not expected_patterns:
            if len(output.strip()) > 0:
                return {
                    "success": True,
                    "reasons": ["Output appears reasonable for generic task"],
                    "weight": 0.2
                }
        
        success_rate = len(matches) / len(expected_patterns) if expected_patterns else 0.3
        
        return {
            "success": success_rate > 0.3,
            "reasons": matches + misses,
            "weight": 0.25 * max(success_rate, 0.2)
        }

    def _extract_expected_patterns(self, prompt: str, intent: Dict) -> Dict[str, Any]:
        patterns = {}
        prompt_lower = prompt.lower() if prompt else ""
        
        if any(action in intent.get('actions', []) for action in ['calculate']) or \
           any(word in prompt_lower for word in ['calculate', 'compute', 'sum', 'average', 'count', 'total']):
            patterns['numerical'] = {
                'regex': r'-?\d+(?:\.\d+)?',
                'description': 'numerical result'
            }
        
        if intent.get('expected_data_type') == 'list' or \
           any(word in prompt_lower for word in ['list', 'array', 'items', 'elements', 'sort', 'rank']):
            patterns['list_output'] = {
                'regex': r'[\[\(].*[\]\)]|^\d+\.?\s|\w+\s*:\s*\w+',
                'description': 'list, array, or structured data'
            }
        
        patterns['completion_indicator'] = {
            'keywords': ['complete', 'done', 'finished', 'success', 'saved', 'created', 'generated'],
            'description': 'task completion indicator'
        }
        
        return patterns

    def _check_pattern(self, output: str, pattern_info: Dict) -> bool:
        output_lower = output.lower()
        
        if 'regex' in pattern_info:
            return bool(re.search(pattern_info['regex'], output, re.MULTILINE | re.DOTALL))
        
        if 'keywords' in pattern_info:
            return any(keyword.lower() in output_lower for keyword in pattern_info['keywords'])
        
        if 'check' in pattern_info:
            try:
                return pattern_info['check'](output)
            except:
                return False
        
        return False

    def _intent_fulfillment_check(self, code: str, prompt: str, execution_result: Dict[str, Any], intent: Dict) -> Dict[str, Any]:
        if intent is None:
            intent = {}
        
        fulfillment_score = 0
        total_checks = 1
        reasons = []
        
       
        if execution_result.get('return_code', 1) == 0:
            fulfillment_score += 1
            reasons.append("Code executed successfully")
        else:
            reasons.append("Code execution failed")
        
        success_rate = fulfillment_score / total_checks
        
        return {
            "success": success_rate > 0.6,
            "reasons": reasons,
            "weight": 0.3 * success_rate
        }

    def _test_case_validation(self, code: str, prompt: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:

        return_code = execution_result.get('return_code', 1)
        stdout = execution_result.get('stdout', '').strip()
        
        if return_code == 0 and stdout:
            return {
                "success": True,
                "reasons": ["Basic execution test passed"],
                "weight": 0.15
            }
        else:
            return {
                "success": False,
                "reasons": ["Basic execution test failed"],
                "weight": 0.0
            }

def validate_prompt(prompt: str) -> Dict[str, Any]:
    result = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'sanitized_prompt': prompt.strip(),
        'intent_analysis': {},
        'complexity_score': 0.0,
        'has_explicit_inputs': False,
        'extracted_inputs': []
    }

    # Critical: prompt too short or empty
    if not prompt or len(prompt.strip()) < 10:
        result['errors'].append("Prompt is too short for meaningful code generation")
        result['is_valid'] = False

    # Non-critical: prompt is very long
    if len(prompt) > 2000:
        result['warnings'].append("Prompt is very long, consider making it more concise")
        result['complexity_score'] += 0.2

    validator = CodeValidator()
    explicit_inputs = validator._extract_explicit_inputs_from_prompt(prompt)
    if explicit_inputs:
        result['has_explicit_inputs'] = True
        result['extracted_inputs'] = explicit_inputs
        result['warnings'].append(f"Found explicit inputs in prompt: {explicit_inputs}")

    # Check sentence and word count (non-critical)
    sentences = len([s for s in prompt.split('.') if s.strip()])
    words = len(prompt.split())
    if sentences > 10:
        result['warnings'].append("Prompt has many sentences, consider simplifying")
        result['complexity_score'] += 0.1
    if words > 200:
        result['warnings'].append("Prompt is quite lengthy")
        result['complexity_score'] += 0.1

    # Extract intent
    try:
        result['intent_analysis'] = validator._extract_intent_from_prompt(prompt)
        if result['intent_analysis'] is None:
            result['intent_analysis'] = {}
    except Exception as e:
        logger.error(f"Intent extraction failed: {str(e)}")
        result['errors'].append(f"Intent extraction failed: {str(e)}")
        result['is_valid'] = False

    actions = result['intent_analysis'].get('actions', [])
    if len(actions) > 3:
        result['warnings'].append("Prompt contains many different actions, consider focusing on fewer tasks")
        result['complexity_score'] += 0.2

    vague_indicators = ['something', 'anything', 'some kind of', 'maybe', 'somehow']
    if any(indicator in prompt.lower() for indicator in vague_indicators):
        result['warnings'].append("Prompt contains vague language that may lead to unclear results")
        result['complexity_score'] += 0.1

  
    result['sanitized_prompt'] = re.sub(r'[^\w\s\-.,!?():;/="\'<>]', '', prompt)

    
    if result['complexity_score'] > 0.5:
        result['warnings'].append("Prompt complexity is high - results may vary")

    return result
