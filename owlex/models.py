"""
Data models for owlex - Task management and API responses.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorCode(str, Enum):
    """Standard error codes for programmatic error handling."""
    INVALID_ARGS = "INVALID_ARGS"
    NOT_FOUND = "NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    CANCELLED = "CANCELLED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class Agent(str, Enum):
    """Available AI agents."""
    CODEX = "codex"
    GEMINI = "gemini"
    OPENCODE = "opencode"
    CLAUDEOR = "claudeor"  # Claude Code via OpenRouter
    AICHAT = "aichat"  # aichat multi-provider CLI


@dataclass
class Task:
    """Represents a background task for CLI execution."""
    task_id: str
    status: str  # TaskStatus value
    command: str
    args: dict
    start_time: datetime
    context: Any | None = field(default=None, repr=False)  # MCP Context
    completion_time: datetime | None = None
    result: str | None = None
    error: str | None = None
    warnings: str | None = None  # stderr output captured even on success
    raw_output: str | None = field(default=None, repr=False)  # stdout before cleaning, for session ID parsing
    async_task: asyncio.Task | None = field(default=None, repr=False)
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    # Streaming support
    output_lines: list[str] = field(default_factory=list)
    stream_complete: bool = False


# === Pydantic Response Models ===

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    error_code: str | None = None
    details: dict[str, Any] | None = None


class TaskResponse(BaseModel):
    """Response for task operations (start, get, wait)."""
    success: bool
    task_id: str | None = None
    status: str | None = None
    message: str | None = None
    content: str | None = None
    warnings: str | None = None  # stderr captured even on success
    error: str | None = None
    error_code: str | None = None  # Standard error code for programmatic handling
    duration_seconds: float | None = None


class AgentResponse(BaseModel):
    """Response from a single agent in council."""
    agent: str
    status: str
    content: str | None = None
    error: str | None = None
    duration_seconds: float | None = None
    task_id: str
    session_id: str | None = None  # Explicit session ID for resume (Option A)


class ClaudeOpinion(BaseModel):
    """Claude's initial opinion provided before council deliberation."""
    content: str
    provided_at: str  # ISO timestamp


class CouncilRound(BaseModel):
    """A single round of council deliberation."""
    codex: AgentResponse | None = None
    gemini: AgentResponse | None = None
    opencode: AgentResponse | None = None
    claudeor: AgentResponse | None = None  # Claude Code via OpenRouter
    aichat: AgentResponse | None = None  # aichat multi-provider CLI


class CouncilMetadata(BaseModel):
    """Metadata for council session."""
    total_duration_seconds: float
    rounds: int
    log: list[str] = Field(default_factory=list)  # Progress log entries


class CouncilResponse(BaseModel):
    """Structured response from council_ask."""
    prompt: str
    working_directory: str | None = None
    deliberation: bool
    critique: bool = False  # If true, round 2 used critique mode instead of revision
    claude_opinion: ClaudeOpinion | None = None
    round_1: CouncilRound
    round_2: CouncilRound | None = None
    # Role assignments (None if all agents used neutral/no roles)
    # Maps agent name to role ID, e.g., {"codex": "security", "gemini": "perf"}
    roles: dict[str, str] | None = None
    metadata: CouncilMetadata
