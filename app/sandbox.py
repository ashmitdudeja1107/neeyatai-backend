import subprocess
import tempfile
import os
import time
import asyncio
from typing import Tuple, Optional, Dict, Any
import logging
from app.config import settings

try:
    from daytona import AsyncDaytona, DaytonaConfig, DaytonaError
    from daytona.common.daytona import CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams
    DAYTONA_SDK_AVAILABLE = True
except ImportError:
    DAYTONA_SDK_AVAILABLE = False
    AsyncDaytona = None
    DaytonaConfig = None
    DaytonaError = Exception
    CreateSandboxFromImageParams = None
    CreateSandboxFromSnapshotParams = None

logger = logging.getLogger(__name__)


class DaytonaSandbox:
    def __init__(self):
        self.daytona_url = settings.daytona_api_url
        self.api_key = settings.daytona_api_key
        self.timeout = settings.execution_timeout
        self._daytona_client = None
        self._active_sandboxes: Dict[str, Any] = {}

    async def execute_code(self, code: str, session_id: str) -> Tuple[str, str, int, float]:
        start_time = time.time()

        try:
            if DAYTONA_SDK_AVAILABLE and self.api_key and await self._check_daytona_connection():
                return await self._execute_in_daytona(code, session_id)
            else:
                logger.warning("Daytona SDK not available or not configured, falling back to local execution")
                return await self._execute_locally(code)

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Code execution failed: {str(e)}")
            return "", f"Execution error: {str(e)}", 1, execution_time

    async def _execute_in_daytona(self, code: str, session_id: str) -> Tuple[str, str, int, float]:
        start_time = time.time()

        try:
            daytona_client = await self._get_daytona_client()

            sandbox = await self._get_or_create_sandbox(daytona_client, session_id)

            temp_filename = f"code_{int(time.time() * 1000)}.py"

            await sandbox.fs.upload_file(code.encode('utf-8'), temp_filename)

            response = await sandbox.process.exec(
                f"python3 {temp_filename}",
                timeout=self.timeout
            )

            execution_time = time.time() - start_time

            try:
                await sandbox.fs.delete_file(temp_filename)
            except Exception:
                pass

            stdout = ""
            stderr = ""
            exit_code = 0
            
            if hasattr(response, 'exit_code'):
                exit_code = response.exit_code or 0
            elif hasattr(response, 'code'):
                exit_code = response.code or 0
            
            if hasattr(response, 'result'):
                result = response.result
                if isinstance(result, str):
                    stdout = result
                elif isinstance(result, dict):
                    stdout = result.get('stdout', '') or result.get('output', '')
                    stderr = result.get('stderr', '') or result.get('error', '')
                elif hasattr(result, 'stdout'):
                    stdout = result.stdout or ""
                    stderr = getattr(result, 'stderr', '') or ""
            
            if not stdout:
                if hasattr(response, 'stdout'):
                    stdout = response.stdout or ""
                elif hasattr(response, 'output'):
                    stdout = response.output or ""
            
            if not stderr:
                if hasattr(response, 'stderr'):
                    stderr = response.stderr or ""
                elif hasattr(response, 'error'):
                    stderr = response.error or ""

            return (stdout, stderr, exit_code, execution_time)

        except DaytonaError as e:
            execution_time = time.time() - start_time
            logger.error(f"Daytona SDK error: {str(e)}")
            return await self._execute_locally(code)
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Daytona execution error: {str(e)}")
            return await self._execute_locally(code)

    async def _get_daytona_client(self) -> Any:
        if self._daytona_client is None:
            config = DaytonaConfig(
                api_key=self.api_key,
                api_url=self.daytona_url,
                target="us"
            )
            self._daytona_client = AsyncDaytona(config)

        return self._daytona_client

    async def _get_or_create_sandbox(self, daytona_client: Any, session_id: str) -> Any:

        if session_id in self._active_sandboxes:
            try:
                sandbox = self._active_sandboxes[session_id]
                await sandbox.process.exec("echo 'ping'", timeout=5)
                return sandbox
            except Exception as e:
                logger.warning(f"Existing sandbox for {session_id} is not responsive: {str(e)}")
                del self._active_sandboxes[session_id]

        try:
            logger.info(f"Creating new Daytona sandbox for session: {session_id}")

            params = CreateSandboxFromImageParams(
                image="python:3.11-slim",
                name=f"session-{session_id}"
            )
            
            sandbox = await daytona_client.create(params, timeout=60)
            
            if sandbox is None:
                raise Exception("Failed to create sandbox")

            self._active_sandboxes[session_id] = sandbox
            logger.info(f"Successfully created sandbox {sandbox.id} for session: {session_id}")
            return sandbox

        except DaytonaError as e:
            logger.error(f"Failed to create Daytona sandbox: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating sandbox: {str(e)}")
            raise

    async def _execute_locally(self, code: str) -> Tuple[str, str, int, float]:
        start_time = time.time()

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            try:
                process = await asyncio.create_subprocess_exec(
                    'python', temp_file_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=tempfile.gettempdir(),
                    env=self._get_safe_env()
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout
                    )

                    execution_time = time.time() - start_time

                    return (
                        stdout.decode('utf-8', errors='ignore')[:settings.max_output_length],
                        stderr.decode('utf-8', errors='ignore')[:settings.max_output_length],
                        process.returncode or 0,
                        execution_time
                    )

                except asyncio.TimeoutError:
                    try:
                        process.kill()
                        await process.wait()
                    except Exception:
                        pass

                    execution_time = time.time() - start_time
                    return "", f"Execution timeout ({self.timeout}s)", 1, execution_time

            finally:
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass

        except Exception as e:
            execution_time = time.time() - start_time
            return "", f"Local execution error: {str(e)}", 1, execution_time

    def _get_safe_env(self) -> dict:
        return {
            'PATH': os.environ.get('PATH', ''),
            'PYTHONPATH': '',
            'HOME': tempfile.gettempdir(),
            'TEMP': tempfile.gettempdir(),
            'TMP': tempfile.gettempdir(),
        }

    async def _check_daytona_connection(self) -> bool:
        if not DAYTONA_SDK_AVAILABLE:
            logger.warning("Daytona SDK is not installed")
            return False

        if not self.api_key:
            logger.warning("Daytona API key not configured")
            return False

        try:
            client = await self._get_daytona_client()
            if hasattr(client, 'list'):
                await client.list()
                return True
            else:
                return True
        except Exception as e:
            logger.error(f"Daytona connection check failed: {str(e)}")
            return False

    async def cleanup_workspace(self, session_id: str) -> bool:
        try:
            if session_id in self._active_sandboxes:
                sandbox = self._active_sandboxes[session_id]
                
                client = await self._get_daytona_client()
                
                try:
                    if hasattr(sandbox, 'id'):
                        if hasattr(client, 'delete') and callable(getattr(client, 'delete')):
                            await client.delete(sandbox)
                            logger.info(f"Successfully deleted sandbox {sandbox.id} for session: {session_id}")
                        else:
                            logger.warning(f"Client delete method not available or not callable")
                    else:
                        logger.warning(f"Sandbox for session {session_id} has no ID attribute")
                        
                except Exception as e:
                    logger.warning(f"Error deleting sandbox: {str(e)}")
                    try:
                        if hasattr(client, 'stop') and callable(getattr(client, 'stop')):
                            await client.stop(sandbox)
                            logger.info(f"Successfully stopped sandbox {sandbox.id}")
                        else:
                            logger.warning(f"Client stop method not available or not callable")
                    except Exception as stop_error:
                        logger.warning(f"Could not stop sandbox: {stop_error}")

                del self._active_sandboxes[session_id]

            return True

        except Exception as e:
            logger.error(f"Workspace cleanup failed for {session_id}: {str(e)}")
            if session_id in self._active_sandboxes:
                del self._active_sandboxes[session_id]
            return False

    async def close(self):
        try:
            for session_id in list(self._active_sandboxes.keys()):
                await self.cleanup_workspace(session_id)

            if self._daytona_client:
                await self._daytona_client.close()
                self._daytona_client = None

        except Exception as e:
            logger.error(f"Error closing Daytona client: {str(e)}")

    def __del__(self):
        if self._daytona_client and hasattr(self._daytona_client, 'close'):
            try:
                asyncio.create_task(self.close())
            except Exception:
                pass