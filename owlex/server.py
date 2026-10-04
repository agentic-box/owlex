#!/usr/bin/env python3
"""
MCP Server for Codex CLI and Gemini CLI Integration
Allows Claude Code to start/resume sessions with Codex or Gemini for advice
"""

import asyncio
import json
import os
import sys
from datetime import datetime

from pydantic import Field
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

from .models import TaskResponse, ErrorCode, Agent
from .engine import engine, resolve_executable, DEFAULT_TIMEOUT, codex_runner, gemini_runner, opencode_runner, claudeor_runner, aichat_runner
from .council import Council
from .config import config
from .roles import get_resolver


# Initialize FastMCP server
mcp = FastMCP("owlex-server")


# === Resources ===

async def _get_cli_version(cmd: str) -> str:
    """Get version string from a CLI tool."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *resolve_executable([cmd, "--version"]),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return stdout.decode().strip().split('\n')[0]
    except Exception:
        return "unknown"


def _get_codex_model() -> str:
    """Get Codex model from config file."""
    import pathlib
    config_path = pathlib.Path.home() / ".codex" / "config.toml"
    try:
        if config_path.exists():
            content = config_path.read_text()
            for line in content.split('\n'):
                if line.startswith('model ='):
                    return line.split('=')[1].strip().strip('"\'')
    except Exception:
        pass
    return "default"


def _get_gemini_model() -> str:
    """Get Gemini model - uses CLI default."""
    # Gemini CLI uses default model based on version
    return "gemini-2.5-pro"  # Default for current CLI


def _get_opencode_model() -> str:
    """Get OpenCode model from env or default."""
    model = os.environ.get("OPENCODE_MODEL", "").strip()
    return model if model else "openrouter/anthropic/claude-sonnet-4"


def _get_aichat_model() -> str:
    """Get aichat model from env or default."""
    model = os.environ.get("AICHAT_MODEL", "").strip()
    return model if model else "default"


@mcp.resource("owlex://agents")
async def get_agents() -> str:
    """List available agents and their configuration."""
    excluded = config.council.exclude_agents

    # Query CLI versions in parallel
    codex_ver, gemini_ver, opencode_ver, aichat_ver = await asyncio.gather(
        _get_cli_version("codex"),
        _get_cli_version("gemini"),
        _get_cli_version("opencode"),
        _get_cli_version("aichat"),
    )

    agents = {
        "codex": {
            "available": "codex" not in excluded,
            "cli_version": codex_ver,
            "model": _get_codex_model(),
            "description": "Deep reasoning, code review, bug finding",
            "config": {
                "enable_search": config.codex.enable_search,
                "bypass_approvals": config.codex.bypass_approvals,
            }
        },
        "gemini": {
            "available": "gemini" not in excluded,
            "cli_version": gemini_ver,
            "model": _get_gemini_model(),
            "description": "1M context window, multimodal, large codebases",
            "config": {
                "yolo_mode": config.gemini.yolo_mode,
            }
        },
        "opencode": {
            "available": "opencode" not in excluded,
            "cli_version": opencode_ver,
            "model": _get_opencode_model(),
            "description": "Alternative perspective, configurable models",
            "config": {
                "agent_mode": config.opencode.agent,
            }
        },
        "aichat": {
            "available": "aichat" not in excluded,
            "cli_version": aichat_ver,
            "model": _get_aichat_model(),
            "description": "Multi-provider LLM CLI, bring your own model",
            "config": {
                "model": config.aichat.model,
            }
        },
    }

    return json.dumps({
        "agents": agents,
        "excluded": list(excluded),
        "default_timeout": config.default_timeout,
    }, indent=2)


@mcp.resource("owlex://council/status")
def get_council_status() -> str:
    """Get status of running council deliberations."""
    council_tasks = []

    for task_id, task in engine.tasks.items():
        if task.command == "council_ask":
            elapsed = (datetime.now() - task.start_time).total_seconds()
            council_tasks.append({
                "task_id": task_id,
                "status": task.status,
                "elapsed_seconds": round(elapsed, 1),
                "prompt": task.args.get("prompt", "")[:100] + "..." if len(task.args.get("prompt", "")) > 100 else task.args.get("prompt", ""),
                "deliberate": task.args.get("deliberate", True),
                "critique": task.args.get("critique", False),
            })

    # Sort by most recent first
    council_tasks.sort(key=lambda x: x["elapsed_seconds"])

    running = [t for t in council_tasks if t["status"] == "running"]
    pending = [t for t in council_tasks if t["status"] == "pending"]

    return json.dumps({
        "running_count": len(running),
        "pending_count": len(pending),
        "total_count": len(council_tasks),
        "running": running,
        "pending": pending,
        "recent": council_tasks[:5],
    }, indent=2)


def _log(msg: str):
    """Log progress to stderr for CLI visibility."""
    print(msg, file=sys.stderr, flush=True)


def _validate_working_directory(working_directory: str | None) -> tuple[str | None, str | None]:
    """Validate and expand working directory. Returns (expanded_path, error_message)."""
    if not working_directory:
        return None, None
    expanded = os.path.expanduser(working_directory)
    if not os.path.isdir(expanded):
        return None, f"working_directory '{working_directory}' does not exist or is not a directory."
    return expanded, None


# === Codex Tools ===

@mcp.tool()
async def start_codex_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for Codex (--cd flag)"),
    enable_search: bool = Field(default=True, description="Enable web search (--search flag)")
) -> dict:
    """Start a new Codex session (no prior context)."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    task = engine.create_task(
        command=f"{Agent.CODEX.value}_exec",
        args={"prompt": prompt.strip(), "working_directory": working_directory, "enable_search": enable_search},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, codex_runner, mode="exec",
        prompt=prompt.strip(), working_directory=working_directory, enable_search=enable_search
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message="Codex session started. Use wait_for_task to get result.",
    ).model_dump()


@mcp.tool()
async def resume_codex_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send to the resumed session"),
    session_id: str | None = Field(default=None, description="Session ID to resume (uses --last if not provided)"),
    working_directory: str | None = Field(default=None, description="Working directory for Codex (--cd flag)"),
    enable_search: bool = Field(default=True, description="Enable web search (--search flag)")
) -> dict:
    """Resume an existing Codex session and ask for advice."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    use_last = not session_id or not session_id.strip()
    session_ref = "--last" if use_last else session_id.strip()

    # Validate session ID if provided (not using --last)
    if not use_last and not codex_runner.validate_session_id(session_ref):
        return TaskResponse(
            success=False,
            error=f"Invalid session_id: '{session_id}' - contains disallowed characters",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    task = engine.create_task(
        command=f"{Agent.CODEX.value}_resume",
        args={"session_id": session_ref, "prompt": prompt.strip(), "working_directory": working_directory, "enable_search": enable_search},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, codex_runner, mode="resume",
        prompt=prompt.strip(), session_ref=session_ref, working_directory=working_directory, enable_search=enable_search
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"Codex resume started{' (last session)' if use_last else f' for session {session_id}'}. Use wait_for_task to get result.",
    ).model_dump()


# === Gemini Tools ===

@mcp.tool()
async def start_gemini_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for Gemini context"),
) -> dict:
    """Start a new Gemini CLI session (no prior context)."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    task = engine.create_task(
        command=f"{Agent.GEMINI.value}_exec",
        args={"prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, gemini_runner, mode="exec",
        prompt=prompt.strip(), working_directory=working_directory
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message="Gemini session started. Use wait_for_task to get result.",
    ).model_dump()


@mcp.tool()
async def resume_gemini_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send to the resumed session"),
    session_ref: str = Field(default="latest", description="Session to resume: 'latest' for most recent, or index number"),
    working_directory: str | None = Field(default=None, description="Working directory for Gemini context"),
) -> dict:
    """Resume an existing Gemini CLI session with full conversation history."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    # Validate session reference (must be numeric index or "latest")
    if not gemini_runner.validate_session_id(session_ref):
        return TaskResponse(
            success=False,
            error=f"Invalid session_ref: '{session_ref}' - must be 'latest' or a numeric index",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    task = engine.create_task(
        command=f"{Agent.GEMINI.value}_resume",
        args={"session_ref": session_ref, "prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, gemini_runner, mode="resume",
        prompt=prompt.strip(), session_ref=session_ref, working_directory=working_directory
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"Gemini resume started (session: {session_ref}). Use wait_for_task to get result.",
    ).model_dump()


# === OpenCode Tools ===

@mcp.tool()
async def start_opencode_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for OpenCode context"),
) -> dict:
    """Start a new OpenCode session (no prior context)."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    task = engine.create_task(
        command=f"{Agent.OPENCODE.value}_exec",
        args={"prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, opencode_runner, mode="exec",
        prompt=prompt.strip(), working_directory=working_directory
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message="OpenCode session started. Use wait_for_task to get result.",
    ).model_dump()


@mcp.tool()
async def resume_opencode_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send to the resumed session"),
    session_id: str | None = Field(default=None, description="Session ID to resume (uses --continue if not provided)"),
    working_directory: str | None = Field(default=None, description="Working directory for OpenCode context"),
) -> dict:
    """Resume an existing OpenCode session with full conversation history."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    use_continue = not session_id or not session_id.strip()
    session_ref = "--continue" if use_continue else session_id.strip()

    # Validate session ID if provided (not using --continue)
    if not use_continue and not opencode_runner.validate_session_id(session_ref):
        return TaskResponse(
            success=False,
            error=f"Invalid session_id: '{session_id}' - contains disallowed characters",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    task = engine.create_task(
        command=f"{Agent.OPENCODE.value}_resume",
        args={"session_id": session_ref, "prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, opencode_runner, mode="resume",
        prompt=prompt.strip(), session_ref=session_ref, working_directory=working_directory
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"OpenCode resume started{' (continuing last session)' if use_continue else f' for session {session_id}'}. Use wait_for_task to get result.",
    ).model_dump()


# === Claude OpenRouter Session Tools ===

@mcp.tool()
async def start_claudeor_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for Claude context"),
) -> dict:
    """
    Start a new Claude Code session via OpenRouter.

    Uses Claude CLI with OpenRouter backend, allowing alternative models
    like DeepSeek, GPT-4o, Gemini, etc. Configure model via CLAUDEOR_MODEL env var.
    """
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    # Check if API key is configured
    if not config.claudeor.api_key:
        return TaskResponse(
            success=False,
            error="OPENROUTER_API_KEY or CLAUDEOR_API_KEY environment variable not set",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    task = engine.create_task(
        command=f"{Agent.CLAUDEOR.value}_exec",
        args={"prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, claudeor_runner, mode="exec",
        prompt=prompt.strip(), working_directory=working_directory
    ))

    model_info = f" ({config.claudeor.model})" if config.claudeor.model else ""
    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"Claude OpenRouter{model_info} session started. Use wait_for_task to get result.",
    ).model_dump()


@mcp.tool()
async def resume_claudeor_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send to the resumed session"),
    session_id: str | None = Field(default=None, description="Session ID to resume (uses --continue if not provided)"),
    working_directory: str | None = Field(default=None, description="Working directory for Claude context"),
) -> dict:
    """Resume an existing Claude OpenRouter session with full conversation history."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    # Check if API key is configured
    if not config.claudeor.api_key:
        return TaskResponse(
            success=False,
            error="OPENROUTER_API_KEY or CLAUDEOR_API_KEY environment variable not set",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    use_continue = not session_id or not session_id.strip()
    session_ref = "continue" if use_continue else session_id.strip()

    # Validate session ID if provided
    if not use_continue and not claudeor_runner.validate_session_id(session_ref):
        return TaskResponse(
            success=False,
            error=f"Invalid session_id: '{session_id}' - contains disallowed characters",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    task = engine.create_task(
        command=f"{Agent.CLAUDEOR.value}_resume",
        args={"session_id": session_ref, "prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, claudeor_runner, mode="resume",
        prompt=prompt.strip(), session_ref=session_ref, working_directory=working_directory
    ))

    model_info = f" ({config.claudeor.model})" if config.claudeor.model else ""
    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"Claude OpenRouter{model_info} resume started{' (continuing last session)' if use_continue else f' for session {session_id}'}. Use wait_for_task to get result.",
    ).model_dump()


# === AiChat Tools ===

@mcp.tool()
async def start_aichat_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for aichat context"),
) -> dict:
    """
    Start a new aichat session.

    Uses the aichat CLI for multi-provider LLM access. Configure model via
    AICHAT_MODEL env var or aichat's own config file (~/.config/aichat/config.yaml).
    """
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    task = engine.create_task(
        command=f"{Agent.AICHAT.value}_exec",
        args={"prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, aichat_runner, mode="exec",
        prompt=prompt.strip(), working_directory=working_directory
    ))

    model_info = f" ({config.aichat.model})" if config.aichat.model else ""
    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"AiChat{model_info} session started. Use wait_for_task to get result.",
    ).model_dump()


@mcp.tool()
async def resume_aichat_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send to the resumed session"),
    session_id: str = Field(description="Session name to resume"),
    working_directory: str | None = Field(default=None, description="Working directory for aichat context"),
) -> dict:
    """Resume an existing aichat session with full conversation history."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    if not session_id or not session_id.strip():
        return TaskResponse(success=False, error="'session_id' parameter is required for aichat resume.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    session_ref = session_id.strip()

    # Validate session ID
    if not aichat_runner.validate_session_id(session_ref):
        return TaskResponse(
            success=False,
            error=f"Invalid session_id: '{session_id}' - contains disallowed characters",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    task = engine.create_task(
        command=f"{Agent.AICHAT.value}_resume",
        args={"session_id": session_ref, "prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, aichat_runner, mode="resume",
        prompt=prompt.strip(), session_ref=session_ref, working_directory=working_directory
    ))

    model_info = f" ({config.aichat.model})" if config.aichat.model else ""
    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"AiChat{model_info} resume started for session {session_id}. Use wait_for_task to get result.",
    ).model_dump()


# === Task Management Tools ===

@mcp.tool()
async def get_task_result(task_id: str) -> dict:
    """
    Get the result of a task (Codex, Gemini, or OpenCode).

    Args:
        task_id: The task ID returned by start/resume session
    """
    task = engine.get_task(task_id)
    if not task:
        return TaskResponse(success=False, error=f"Task '{task_id}' not found.", error_code=ErrorCode.NOT_FOUND).model_dump()

    if task.status == "pending":
        return TaskResponse(
            success=True,
            task_id=task_id,
            status=task.status,
            message="Task is still pending.",
        ).model_dump()
    elif task.status == "running":
        elapsed = (datetime.now() - task.start_time).total_seconds()
        return TaskResponse(
            success=True,
            task_id=task_id,
            status=task.status,
            message=f"Task is still running ({elapsed:.1f}s elapsed).",
        ).model_dump()
    elif task.status == "completed":
        return TaskResponse(
            success=True,
            task_id=task_id,
            status=task.status,
            content=task.result,
            warnings=task.warnings,
            duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
        ).model_dump()
    elif task.status == "failed":
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=task.error,
            error_code=ErrorCode.EXECUTION_FAILED,
            duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
        ).model_dump()
    elif task.status == "cancelled":
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=task.error or "Task was cancelled.",
            error_code=ErrorCode.CANCELLED,
            duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
        ).model_dump()
    else:
        # Unknown status - should not happen but handle gracefully
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=f"Unexpected task status: {task.status}",
            error_code=ErrorCode.INTERNAL_ERROR,
        ).model_dump()


@mcp.tool()
async def wait_for_task(task_id: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """
    Wait for a task to complete and return its result.

    Args:
        task_id: The task ID to wait for
        timeout: Maximum seconds to wait (default: 300)
    """
    task = engine.get_task(task_id)
    if not task:
        return TaskResponse(success=False, error=f"Task '{task_id}' not found.", error_code=ErrorCode.NOT_FOUND).model_dump()

    if task.status in ["completed", "failed", "cancelled"]:
        if task.status == "completed":
            return TaskResponse(
                success=True,
                task_id=task_id,
                status=task.status,
                content=task.result,
                warnings=task.warnings,
                duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
            ).model_dump()
        error_code = ErrorCode.EXECUTION_FAILED if task.status == "failed" else ErrorCode.CANCELLED
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=task.error,
            error_code=error_code,
        ).model_dump()

    if task.async_task:
        # Check if task already completed (e.g., between abort and re-wait)
        if task.async_task.done():
            try:
                task.async_task.result()  # Re-raise any exception from the task
            except asyncio.CancelledError:
                # Task was cancelled - mark appropriately
                if task.status not in ["completed", "failed", "cancelled"]:
                    task.status = "cancelled"
                    task.error = "Task was cancelled"
                    task.completion_time = datetime.now()
            except BaseException as e:
                # Catch BaseException to handle all exceptions including SystemExit, KeyboardInterrupt
                if task.status not in ["completed", "failed", "cancelled"]:
                    task.status = "failed"
                    task.error = f"Task failed: {str(e)}"
                    task.completion_time = datetime.now()
            # Fall through to return result below
        else:
            try:
                await asyncio.wait_for(asyncio.shield(task.async_task), timeout=timeout)
            except asyncio.TimeoutError:
                return TaskResponse(
                    success=False,
                    task_id=task_id,
                    status="timeout",
                    error=f"Task still running after {timeout}s. Use get_task_result to check later.",
                    error_code=ErrorCode.TIMEOUT,
                ).model_dump()
            except asyncio.CancelledError:
                # User aborted the wait (e.g., pressed ESC) - task keeps running
                return TaskResponse(
                    success=True,
                    task_id=task_id,
                    status=task.status,
                    message="Wait aborted. Task still running. Use get_task_result or wait_for_task later.",
                ).model_dump()
            except Exception as e:
                # Bug fix: Set task.status and task.error so subsequent calls are consistent
                task.status = "failed"
                task.error = f"Task failed: {str(e)}"
                task.completion_time = datetime.now()
                return TaskResponse(
                    success=False,
                    task_id=task_id,
                    status=task.status,
                    error=task.error,
                    error_code=ErrorCode.INTERNAL_ERROR,
                ).model_dump()
    else:
        # Bug fix: No async_task means task may have failed before launch or is in unexpected state
        # Return actual status instead of assuming CANCELLED
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=task.error or f"Task has no async handler (status: {task.status})",
            error_code=ErrorCode.INTERNAL_ERROR,
        ).model_dump()

    if task.status == "completed":
        return TaskResponse(
            success=True,
            task_id=task_id,
            status=task.status,
            content=task.result,
            warnings=task.warnings,
            duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
        ).model_dump()

    error_code = ErrorCode.EXECUTION_FAILED if task.status == "failed" else ErrorCode.CANCELLED
    return TaskResponse(
        success=False,
        task_id=task_id,
        status=task.status,
        error=task.error,
        error_code=error_code,
    ).model_dump()


@mcp.tool()
async def list_tasks(
    status_filter: str | None = Field(default=None, description="Filter by status: pending, running, completed, failed, cancelled"),
    limit: int = Field(default=20, description="Maximum number of tasks to return"),
) -> dict:
    """
    List all tracked tasks with their current status.

    Args:
        status_filter: Optional filter by task status
        limit: Maximum number of tasks to return (default: 20)
    """
    tasks_list = []
    for task_id, task in list(engine.tasks.items())[-limit:]:
        if status_filter and task.status != status_filter:
            continue
        elapsed = (datetime.now() - task.start_time).total_seconds()
        tasks_list.append({
            "task_id": task_id,
            "command": task.command,
            "status": task.status,
            "elapsed_seconds": round(elapsed, 1),
            "has_result": task.result is not None,
            "has_error": task.error is not None,
        })

    return {
        "success": True,
        "count": len(tasks_list),
        "tasks": tasks_list,
    }


@mcp.tool()
async def cancel_task(task_id: str) -> dict:
    """
    Cancel a running task and kill its subprocess.

    Args:
        task_id: The task ID to cancel
    """
    task = engine.get_task(task_id)
    if not task:
        return TaskResponse(success=False, error=f"Task '{task_id}' not found.", error_code=ErrorCode.NOT_FOUND).model_dump()

    if task.status in ["completed", "failed", "cancelled"]:
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=f"Task already {task.status}, cannot cancel.",
            error_code=ErrorCode.INVALID_ARGS,
        ).model_dump()

    # Kill the subprocess and cancel the async task
    await engine.kill_task_subprocess(task)
    task.status = "cancelled"
    task.error = "Cancelled by user"
    task.completion_time = datetime.now()

    return TaskResponse(
        success=True,
        task_id=task_id,
        status=task.status,
        message="Task cancelled successfully.",
    ).model_dump()


# === Council Tool ===

async def _run_council_deliberation(
    task,
    prompt: str,
    working_directory: str | None,
    claude_opinion: str | None,
    deliberate: bool,
    critique: bool,
    timeout: int,
    roles: dict[str, str] | list[str] | None = None,
    team: str | None = None,
):
    """Run council deliberation and update task with result."""
    from .models import TaskStatus
    try:
        task.status = TaskStatus.RUNNING.value
        council = Council(context=task.context)
        response = await council.deliberate(
            prompt=prompt,
            working_directory=working_directory,
            claude_opinion=claude_opinion,
            deliberate=deliberate,
            critique=critique,
            timeout=timeout,
            roles=roles,
            team=team,
        )
        task.result = response.model_dump_json(indent=2)
        task.status = TaskStatus.COMPLETED.value
        task.end_time = datetime.now()
    except ValueError as e:
        # Role/team validation errors
        task.error = str(e)
        task.status = TaskStatus.FAILED.value
        task.end_time = datetime.now()
    except Exception as e:
        task.error = str(e)
        task.status = TaskStatus.FAILED.value
        task.end_time = datetime.now()


@mcp.tool()
async def council_ask(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or task to send to the council"),
    claude_opinion: str | None = Field(default=None, description="Claude's initial opinion to share with the council"),
    working_directory: str | None = Field(default=None, description="Working directory for context"),
    deliberate: bool = Field(default=True, description="If true, share answers between agents for a second round of deliberation"),
    critique: bool = Field(default=False, description="If true, round 2 asks agents to critique/find flaws instead of revise"),
    timeout: int = Field(default=DEFAULT_TIMEOUT, description="Timeout per agent in seconds"),
    roles: dict[str, str] | list[str] | None = Field(
        default=None,
        description=(
            "Role assignments for agents. Can be:\n"
            "- Dict mapping agent to role: {\"codex\": \"security\", \"gemini\": \"perf\"}\n"
            "- List of roles (auto-assigned in order): [\"security\", \"perf\", \"skeptic\"]\n"
            "Built-in roles: security, perf, skeptic, architect, maintainer, dx, testing"
        )
    ),
    team: str | None = Field(
        default=None,
        description=(
            "Team preset name (alternative to roles). "
            "Built-in teams: security_audit, code_review, architecture_review, devil_advocate, balanced"
        )
    ),
) -> dict:
    """
    Ask the council (Codex, Gemini, and OpenCode) a question and collect their answers.

    Sends the prompt to all three agents in parallel, waits for responses,
    and returns all answers for the MCP client (Claude Code) to synthesize.

    Supports specialist roles ("hats") for agents to operate with specific perspectives.

    Role specification (mutually exclusive, priority order):
    1. roles parameter - explicit mapping or auto-assign list
    2. team parameter - use a predefined team preset
    3. Neither - all agents operate without special roles

    Examples:
    - roles={"codex": "security", "gemini": "perf"} - explicit assignment
    - roles=["security", "perf", "skeptic"] - auto-assign to codex, gemini, opencode
    - team="security_audit" - use the security audit team preset

    If claude_opinion is provided, it will be shared with other council members
    during deliberation so they can consider Claude's perspective.

    If deliberate=True, shares all answers (including Claude's) with each agent
    for a second round, allowing them to revise after seeing others' responses.

    If critique=True, round 2 asks agents to find bugs, security issues, and
    architectural flaws instead of politely revising their answers.
    """
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    # Validate that roles and team are not both specified
    if roles is not None and team is not None:
        return TaskResponse(
            success=False,
            error="Cannot specify both 'roles' and 'team' parameters. Use one or the other.",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    # Early validation of roles/team to return proper error codes
    excluded = config.council.exclude_agents
    active_agents = [a for a in ["codex", "gemini", "opencode", "claudeor", "aichat"] if a not in excluded]

    # Use default team from config if no roles/team specified
    effective_team = team if team is not None else config.council.default_team
    role_spec = roles if roles is not None else effective_team
    try:
        resolver = get_resolver()
        resolved_roles = resolver.resolve(role_spec, active_agents)
    except ValueError as e:
        return TaskResponse(
            success=False,
            error=str(e),
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    # Build role summary for initial response
    role_assignments = {agent: role.id for agent, role in resolved_roles.items()}
    role_names = {agent: role.name for agent, role in resolved_roles.items()}

    # Create task and run council deliberation asynchronously
    task = engine.create_task(
        command="council_ask",
        args={
            "prompt": prompt.strip(),
            "working_directory": working_directory,
            "claude_opinion": claude_opinion,
            "deliberate": deliberate,
            "critique": critique,
            "timeout": timeout,
            "roles": roles,
            "team": effective_team,
        },
        context=ctx,
    )

    task.async_task = asyncio.create_task(_run_council_deliberation(
        task,
        prompt=prompt.strip(),
        working_directory=working_directory,
        claude_opinion=claude_opinion,
        deliberate=deliberate,
        critique=critique,
        timeout=timeout,
        roles=roles,
        team=effective_team,
    ))

    # Return with role information
    response = TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message="Council deliberation started. Use wait_for_task to get result.",
    ).model_dump()

    # Add role details to response
    response["council"] = {
        "agents": active_agents,
        "excluded": list(excluded),
        "team": effective_team,
        "roles": role_assignments,
        "role_names": role_names,
        "include_claude_opinion": config.council.include_claude_opinion,
    }

    return response


def main():
    """Entry point for owlex-server command."""
    import argparse
    import signal

    from . import __version__

    parser = argparse.ArgumentParser(
        prog="owlex-server",
        description="MCP server for multi-agent CLI orchestration"
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"owlex {__version__}"
    )
    parser.parse_args()

    async def run_with_cleanup():
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def signal_handler(sig):
            _log(f"Received signal {sig}, shutting down...")
            shutdown_event.set()

        # Register signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        engine.start_cleanup_loop()
        try:
            # Run MCP server with shutdown monitoring
            server_task = asyncio.create_task(mcp.run_stdio_async())
            shutdown_task = asyncio.create_task(shutdown_event.wait())

            done, pending = await asyncio.wait(
                [server_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # If shutdown was triggered, cancel the server
            if shutdown_task in done:
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass
            # If server task completed (possibly due to client disconnect), log it
            if server_task in done:
                try:
                    server_task.result()  # Re-raise any exception
                except Exception as e:
                    _log(f"Server task ended: {e}")
        except asyncio.CancelledError:
            _log("Server cancelled")
        except Exception as e:
            _log(f"Server error: {e}")
        finally:
            # Kill all running tasks before exit
            await engine.kill_all_tasks()
            engine.stop_cleanup_loop()
            _log("Server shutdown complete.")

    asyncio.run(run_with_cleanup())


if __name__ == "__main__":
    main()
