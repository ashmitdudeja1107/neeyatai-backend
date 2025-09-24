import subprocess
import tempfile
import os
import time
import asyncio
from typing import Tuple, Dict, Any
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
        """Execute code strictly in Daytona sandbox. No local fallback."""
        start_time = time.time()

        if not DAYTONA_SDK_AVAILABLE:
            execution_time = time.time() - start_time
            logger.error("Daytona SDK is not available")
            return "", "Daytona SDK not installed", 1, execution_time

        if not self.api_key:
            execution_time = time.time() - start_time
            logger.error("Daytona API key is not configured")
            return "", "Daytona API key missing", 1, execution_time

        if not await self._check_daytona_connection():
            execution_time = time.time() - start_time
            logger.error("Cannot connect to Daytona API")
            return "", "Daytona connection failed", 1, execution_time

        try:
            return await self._execute_in_daytona(code, session_id)
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Sandbox execution failed: {str(e)}")
            return "", f"Sandbox execution failed: {str(e)}", 1, execution_time

    async def _execute_in_daytona(self, code: str, session_id: str) -> Tuple[str, str, int, float]:
        start_time = time.time()
        daytona_client = await self._get_daytona_client()
        sandbox = await self._get_or_create_sandbox(daytona_client, session_id)

        temp_filename = f"code_{int(time.time() * 1000)}.py"
        await sandbox.fs.upload_file(code.encode('utf-8'), temp_filename)

        try:
            response = await sandbox.process.exec(
                f"python3 {temp_filename}",
                timeout=self.timeout
            )

            stdout, stderr, exit_code = "", "", 0

            # Extract results
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

            # Fallback to response attributes if result was empty
            if not stdout:
                stdout = getattr(response, 'stdout', '') or getattr(response, 'output', '')
            if not stderr:
                stderr = getattr(response, 'stderr', '') or getattr(response, 'error', '')

            return stdout, stderr, exit_code, time.time() - start_time

        finally:
            try:
                await sandbox.fs.delete_file(temp_filename)
            except Exception:
                pass

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
            sandbox = self._active_sandboxes[session_id]
            try:
                await sandbox.process.exec("echo 'ping'", timeout=5)
                return sandbox
            except Exception:
                del self._active_sandboxes[session_id]

        logger.info(f"Creating new Daytona sandbox for session: {session_id}")
        params = CreateSandboxFromImageParams(image="python:3.11-slim", name=f"session-{session_id}")
        sandbox = await daytona_client.create(params, timeout=60)

        if sandbox is None:
            raise Exception("Failed to create sandbox")

        self._active_sandboxes[session_id] = sandbox
        logger.info(f"Successfully created sandbox {sandbox.id} for session: {session_id}")
        return sandbox

    async def _check_daytona_connection(self) -> bool:
        try:
            client = await self._get_daytona_client()
            if hasattr(client, 'list'):
                await client.list()
            return True
        except Exception as e:
            logger.error(f"Daytona connection check failed: {str(e)}")
            return False

    async def cleanup_workspace(self, session_id: str) -> bool:
        try:
            if session_id in self._active_sandboxes:
                sandbox = self._active_sandboxes.pop(session_id)
                client = await self._get_daytona_client()
                if hasattr(client, 'delete') and callable(getattr(client, 'delete')):
                    await client.delete(sandbox)
            return True
        except Exception as e:
            logger.error(f"Workspace cleanup failed for {session_id}: {str(e)}")
            return False

    async def close(self):
        for session_id in list(self._active_sandboxes.keys()):
            await self.cleanup_workspace(session_id)
        if self._daytona_client:
            await self._daytona_client.close()
            self._daytona_client = None
