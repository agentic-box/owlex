"""
Codex CLI agent runner.
"""

import asyncio
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from ..config import config
from .base import AgentRunner, AgentCommand


async def get_latest_codex_session(
    since_mtime: float | None = None,
    max_retries: int = 3,
    retry_delay: float = 0.3,
) -> str | None:
    """
    Find the most recent Codex session ID from filesystem.

    Codex stores sessions in ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl
    The UUID is extracted from the filename.

    Args:
        since_mtime: Only consider files modified after this timestamp.
                     Used to scope to sessions created during current run.
        max_retries: Number of retries if no session found (handles I/O lag).
        retry_delay: Delay between retries in seconds.

    Returns:
        Session UUID if found, None otherwise
    """
    codex_dir = Path.home() / ".codex" / "sessions"
    if not codex_dir.exists():
        return None

    for attempt in range(max_retries):
        latest_file: Path | None = None
        latest_mtime: float = 0

        # Check recent date directories (today and yesterday)
        # Use timedelta for correct month/year boundary handling
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        date_dirs = [
            codex_dir / f"{now.year}" / f"{now.month:02d}" / f"{now.day:02d}",
            codex_dir / f"{yesterday.year}" / f"{yesterday.month:02d}" / f"{yesterday.day:02d}",
        ]

        for date_dir in date_dirs:
            if not date_dir.exists():
                continue
            try:
                for session_file in date_dir.glob("rollout-*.jsonl"):
                    try:
                        mtime = session_file.stat().st_mtime
                        # Skip files older than since_mtime if specified
                        if since_mtime is not None and mtime < since_mtime:
                            continue
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_file = session_file
                    except OSError:
                        continue  # File may have been deleted
            except OSError:
                continue  # Directory access error

        if latest_file is not None:
            # Extract UUID from filename
            filename = latest_file.stem
            match = re.search(r'rollout-\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-([a-f0-9-]+)$', filename)
            if match:
                return match.group(1)

        # Retry with delay if no session found (I/O lag)
        # Uses asyncio.sleep to avoid blocking the event loop
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)

    return None


def clean_codex_output(raw_output: str, original_prompt: str = "") -> str:
    """Clean Codex CLI output by removing echoed prompt templates."""
    if not config.codex.clean_output:
        return raw_output
    cleaned = raw_output
    if original_prompt and cleaned.startswith(original_prompt):
        cleaned = cleaned[len(original_prompt):].lstrip()
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


class CodexRunner(AgentRunner):
    """Runner for OpenAI Codex CLI."""

    @property
    def name(self) -> str:
        return "codex"

    def build_exec_command(
        self,
        prompt: str,
        working_directory: str | None = None,
        enable_search: bool = False,
        **kwargs,
    ) -> AgentCommand:
        """Build command for starting a new Codex session."""
        full_command = ["codex", "exec", "--skip-git-repo-check"]

        if working_directory:
            full_command.extend(["--cd", working_directory])
        if enable_search:
            full_command.extend(["--enable", "web_search_request"])

        if config.codex.bypass_approvals:
            full_command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            # Equivalent of the removed --full-auto flag: sandboxed, no approval prompts
            full_command.extend(["--sandbox", config.codex.sandbox])
            full_command.extend(["-c", 'approval_policy="never"'])

        # Use stdin for prompt input
        full_command.append("-")

        return AgentCommand(
            command=full_command,
            prompt=prompt,
            output_prefix="Codex Output",
            not_found_hint="Please ensure Codex CLI is installed and in your PATH.",
            stream=True,
        )

    def build_resume_command(
        self,
        session_ref: str,
        prompt: str,
        working_directory: str | None = None,
        enable_search: bool = False,
        **kwargs,
    ) -> AgentCommand:
        """Build command for resuming an existing Codex session."""
        full_command = ["codex", "exec", "--skip-git-repo-check"]

        if working_directory:
            full_command.extend(["--cd", working_directory])
        if enable_search:
            full_command.extend(["--enable", "web_search_request"])

        if config.codex.bypass_approvals:
            full_command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            # Equivalent of the removed --full-auto flag: sandboxed, no approval prompts
            full_command.extend(["--sandbox", config.codex.sandbox])
            full_command.extend(["-c", 'approval_policy="never"'])

        full_command.append("resume")
        if session_ref == "--last":
            full_command.append("--last")
        else:
            # Validate session_ref to prevent flag injection
            # Session IDs should be alphanumeric/UUID-like
            if session_ref.startswith("-"):
                raise ValueError(f"Invalid session_ref: '{session_ref}' - cannot start with '-'")
            full_command.append(session_ref)

        # Use stdin for prompt input
        full_command.append("-")

        return AgentCommand(
            command=full_command,
            prompt=prompt,
            output_prefix="Codex Resume Output",
            not_found_hint="Please ensure Codex CLI is installed and in your PATH.",
            stream=False,  # Resume uses non-streaming mode
        )

    def get_output_cleaner(self) -> Callable[[str, str], str]:
        return clean_codex_output

    async def parse_session_id(
        self,
        output: str,
        since_mtime: float | None = None,
        working_directory: str | None = None,
    ) -> str | None:
        """
        Get session ID for Codex.

        Codex doesn't output session ID in stdout, so we check the filesystem
        for the most recently created session file.

        Note: Codex sessions are stored globally (not project-scoped), so
        working_directory is ignored. Session scoping relies on since_mtime.

        Args:
            output: Ignored (Codex doesn't output session IDs)
            since_mtime: Only consider sessions created after this timestamp
            working_directory: Ignored (Codex sessions are global)
        """
        return await get_latest_codex_session(since_mtime=since_mtime)
