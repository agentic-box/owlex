"""
Claude Code via OpenRouter agent runner.
Uses Claude CLI with OpenRouter as the backend for alternative models.
"""

import json
import re
from typing import Callable

from ..config import config
from .base import AgentRunner, AgentCommand


def _parse_claude_json(raw_output: str) -> dict | None:
    """Parse the JSON object `claude --output-format json` prints first."""
    try:
        data, _ = json.JSONDecoder().raw_decode((raw_output or "").lstrip())
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def clean_claudeor_output(raw_output: str, original_prompt: str = "") -> str:
    """Extract the answer from Claude's JSON output and tidy it."""
    data = _parse_claude_json(raw_output)
    if data is not None and isinstance(data.get("result"), str):
        text = data["result"]
    else:
        text = raw_output
    if not config.claudeor.clean_output:
        return text
    # Remove excessive newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', text)
    return cleaned.strip()


class ClaudeORRunner(AgentRunner):
    """Runner for Claude Code CLI via OpenRouter."""

    @property
    def name(self) -> str:
        return "claudeor"

    def _get_env_overrides(self) -> dict[str, str]:
        """Get environment variable overrides for OpenRouter."""
        import os

        env = {
            # OpenRouter requires /api not /api/v1
            "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
            # Must set ANTHROPIC_API_KEY to empty to avoid conflicts
            "ANTHROPIC_API_KEY": "",
        }

        # Get API key - check config first, then current environment
        # (env vars from .mcp.json may not be available at config load time)
        api_key = (
            config.claudeor.api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("CLAUDEOR_API_KEY")
        )

        # Use ANTHROPIC_AUTH_TOKEN for OpenRouter API key (not ANTHROPIC_API_KEY)
        if api_key:
            env["ANTHROPIC_AUTH_TOKEN"] = api_key

        # Get model - check config first, then current environment
        model = config.claudeor.model or os.environ.get("CLAUDEOR_MODEL")
        if model:
            env["ANTHROPIC_MODEL"] = model

        return env

    def build_exec_command(
        self,
        prompt: str,
        working_directory: str | None = None,
        enable_search: bool = False,
        **kwargs,
    ) -> AgentCommand:
        """Build command for starting a new Claude OpenRouter session."""
        full_command = ["claude"]

        # JSON output carries the session ID that round 2 resumes
        full_command.extend(["--output-format", "json"])

        # Use print mode for single-shot execution
        full_command.extend(["--print", "-p", prompt])

        # Set working directory if provided
        cwd = working_directory

        return AgentCommand(
            command=full_command,
            prompt="",  # Prompt is in command args
            cwd=cwd,
            output_prefix="Claude (OpenRouter) Output",
            not_found_hint="Please ensure Claude CLI is installed. Run: npm install -g @anthropic-ai/claude-code",
            stream=False,  # Print mode doesn't stream
            env_overrides=self._get_env_overrides(),
        )

    def build_resume_command(
        self,
        session_ref: str,
        prompt: str,
        working_directory: str | None = None,
        enable_search: bool = False,
        **kwargs,
    ) -> AgentCommand:
        """Build command for resuming an existing Claude session."""
        full_command = ["claude"]

        # Resume with session reference
        if session_ref == "--continue" or session_ref == "continue":
            full_command.append("--continue")
        else:
            # Validate session_ref
            if session_ref.startswith("-"):
                raise ValueError(f"Invalid session_ref: '{session_ref}' - cannot start with '-'")
            full_command.extend(["--resume", session_ref])

        full_command.extend(["--output-format", "json"])

        # Use print mode
        full_command.extend(["--print", "-p", prompt])

        cwd = working_directory

        return AgentCommand(
            command=full_command,
            prompt="",  # Prompt is in command args
            cwd=cwd,
            output_prefix="Claude (OpenRouter) Output",
            not_found_hint="Please ensure Claude CLI is installed. Run: npm install -g @anthropic-ai/claude-code",
            stream=False,
            env_overrides=self._get_env_overrides(),
        )

    def get_output_cleaner(self) -> Callable[[str, str], str]:
        return clean_claudeor_output

    async def parse_session_id(
        self,
        output: str,
        since_mtime: float | None = None,
        working_directory: str | None = None,
    ) -> str | None:
        """
        Read the session ID from Claude's JSON output.

        Returns None when it is missing, so round 2 falls back to exec mode.
        Picking the newest transcript under ~/.claude/projects would usually
        select the live session of the Claude Code instance running the
        council.

        Args:
            output: Captured stdout and stderr of the claude run
            since_mtime: Ignored
            working_directory: Ignored
        """
        data = _parse_claude_json(output)
        session_id = data.get("session_id") if data else None
        return session_id if isinstance(session_id, str) else None
