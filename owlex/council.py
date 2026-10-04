"""
Council orchestration logic for multi-agent deliberation.
Handles parallel execution and deliberation rounds.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Any

from .config import config
from .engine import engine, build_agent_response, codex_runner, gemini_runner, opencode_runner, claudeor_runner, aichat_runner
from .prompts import inject_role_prefix, build_deliberation_prompt_with_role
from .roles import RoleSpec, RoleDefinition, RoleResolver, RoleId, get_resolver
from .models import (
    Agent,
    ClaudeOpinion,
    CouncilResponse,
    CouncilRound,
    CouncilMetadata,
    Task,
)


def _log(msg: str):
    """Log progress to stderr for CLI visibility."""
    print(msg, file=sys.stderr, flush=True)


def _captured_output(task: Task) -> str:
    """Raw stdout then stderr of a finished task, for session ID parsing."""
    return "\n".join(part for part in (task.raw_output, task.warnings) if part)


class Council:
    """
    Orchestrates multi-agent deliberation between Codex, Gemini, and OpenCode.

    The council process:
    1. Round 1: All agents answer the question in parallel
    2. Round 2 (optional): Agents see each other's answers and revise/critique

    Supports specialist roles ("hats") for agents to operate with specific perspectives.
    """

    def __init__(
        self,
        context: Any = None,
        task_engine: Any = None,
        role_resolver: RoleResolver | None = None,
    ):
        """
        Initialize the Council.

        Args:
            context: MCP server context for notifications
            task_engine: Optional TaskEngine instance (uses global engine if not provided).
                        This enables dependency injection for testing.
            role_resolver: Optional RoleResolver instance (uses global resolver if not provided).
        """
        self.context = context
        self._engine = task_engine if task_engine is not None else engine
        self._resolver = role_resolver if role_resolver is not None else get_resolver()
        self.log_entries: list[str] = []

    def log(self, msg: str):
        """Add to log and print to stderr."""
        self.log_entries.append(msg)
        _log(msg)

    async def notify(self, message: str, level: str = "info", progress: float | None = None):
        """Send notification to MCP client if context supports it."""
        if not self.context:
            return
        try:
            session = getattr(self.context, 'session', None)
            if not session:
                return

            # Try progress notification (more visible in Claude Code)
            if hasattr(session, 'send_progress_notification') and progress is not None:
                try:
                    await session.send_progress_notification(
                        progress_token="owlex-council",
                        progress=progress,
                        total=100.0,
                        message=message,
                    )
                except Exception:
                    pass

            # Also send as log message
            if hasattr(session, 'send_log_message'):
                await session.send_log_message(level=level, data=message, logger="owlex")
        except Exception:
            pass

    async def deliberate(
        self,
        prompt: str,
        working_directory: str | None = None,
        claude_opinion: str | None = None,
        deliberate: bool = True,
        critique: bool = True,
        timeout: int | None = None,
        roles: RoleSpec = None,
        team: str | None = None,
    ) -> CouncilResponse:
        """
        Run a council deliberation session.

        Args:
            prompt: The question or task to deliberate on
            working_directory: Working directory context for agents (defaults to CWD)
            claude_opinion: Optional Claude opinion to share with agents
            deliberate: If True, run a second round where agents see each other's answers
            critique: If True, Round 2 asks agents to find flaws instead of revise
            timeout: Timeout per agent in seconds
            roles: Role specification - dict, list, or None (see RoleSpec)
            team: Team preset name (alternative to roles parameter)

        Role specification (mutually exclusive):
            - roles: explicit mapping or auto-assign list
            - team: team preset name
            - Neither: all agents neutral

        Returns:
            CouncilResponse with all rounds and metadata

        Raises:
            ValueError: If both roles and team are specified
        """
        # Validate mutual exclusivity of roles/team (consistent with MCP API)
        if roles is not None and team is not None:
            raise ValueError("Cannot specify both 'roles' and 'team' parameters. Use one or the other.")

        if timeout is None:
            timeout = config.default_timeout
        # timeout=0 means no timeout (wait indefinitely)
        effective_timeout = None if timeout == 0 else timeout

        # Default to CWD if no working_directory provided
        if working_directory is None:
            working_directory = os.getcwd()

        # Determine which agents to run
        excluded = config.council.exclude_agents
        # Include claudeor only if API key is configured
        all_agents = ["codex", "gemini", "opencode"]
        if config.claudeor.api_key:
            all_agents.append("claudeor")
        all_agents.append("aichat")
        active_agents = [a for a in all_agents if a not in excluded]

        # Resolve roles (team parameter is just a string that resolves to a team preset)
        role_spec = roles if roles is not None else team
        resolved_roles = self._resolver.resolve(role_spec, active_agents)

        council_start = datetime.now()

        # Log and notify role assignments (skip neutral roles)
        role_msgs = [f"{agent.title()}: {role.name}" for agent, role in resolved_roles.items() if role.id != RoleId.NEUTRAL.value]
        if role_msgs:
            roles_summary = ", ".join(role_msgs)
            self.log(f"Roles assigned: {roles_summary}")
            await self.notify(f"Council roles: {roles_summary}", progress=10)

        # === Round 1: Parallel initial queries ===
        if claude_opinion and claude_opinion.strip():
            self.log(f"Claude's opinion received ({len(claude_opinion)} chars)")

        self.log(f"Round 1: querying {', '.join(a.title() for a in active_agents)}...")
        await self.notify(f"Council Round 1: querying {', '.join(a.title() for a in active_agents)}", progress=20)

        round_1 = await self._run_round_1(prompt, working_directory, effective_timeout, resolved_roles)
        await self.notify("Council Round 1 complete", progress=50)

        round_2 = None
        if deliberate:
            await self.notify("Council Round 2: deliberation phase", progress=60)
            round_2 = await self._run_round_2(
                prompt=prompt,
                working_directory=working_directory,
                round_1=round_1,
                claude_opinion=claude_opinion,
                critique=critique,
                timeout=effective_timeout,
                roles=resolved_roles,
            )

        # Build Claude opinion object if provided
        claude_opinion_obj = None
        if claude_opinion and claude_opinion.strip():
            claude_opinion_obj = ClaudeOpinion(
                content=claude_opinion.strip(),
                provided_at=council_start.isoformat(),
            )

        total_duration = (datetime.now() - council_start).total_seconds()
        await self.notify(f"Council deliberation complete ({total_duration:.1f}s)", progress=100)

        return CouncilResponse(
            prompt=prompt,
            working_directory=working_directory,
            deliberation=deliberate,
            critique=critique,
            claude_opinion=claude_opinion_obj,
            round_1=round_1,
            round_2=round_2,
            roles=self._build_role_assignments(resolved_roles),
            metadata=CouncilMetadata(
                total_duration_seconds=(datetime.now() - council_start).total_seconds(),
                rounds=2 if deliberate else 1,
                log=self.log_entries,
            ),
        )

    def _build_role_assignments(
        self,
        resolved_roles: dict[str, RoleDefinition],
    ) -> dict[str, str] | None:
        """Build role assignments dict for response (excludes neutral roles)."""
        assignments = {
            agent: role.id
            for agent, role in resolved_roles.items()
            if role.id != RoleId.NEUTRAL.value
        }
        return assignments if assignments else None

    async def _run_round_1(
        self,
        prompt: str,
        working_directory: str | None,
        timeout: int | None,
        roles: dict[str, RoleDefinition],
    ) -> CouncilRound:
        """Run the first round of parallel queries.

        Option A: R1 uses exec mode (clean start) for all agents.
        Session IDs are captured after completion for use in R2.
        Role prefixes are injected into prompts based on resolved roles.
        """
        round1_start = datetime.now()
        excluded = config.council.exclude_agents

        tasks = {}
        async_tasks = []

        # Create tasks for non-excluded agents
        # All R1 agents use exec mode (clean start) per Option A
        # Role prefixes are injected into prompts
        if "codex" not in excluded:
            codex_role = roles.get("codex")
            codex_prompt = inject_role_prefix(prompt, codex_role)

            codex_task = self._engine.create_task(
                command=f"council_{Agent.CODEX.value}",
                args={"prompt": codex_prompt, "working_directory": working_directory},
                context=self.context,
            )
            tasks["codex"] = codex_task

            # Capture prompt in closure
            _codex_prompt = codex_prompt

            async def run_codex():
                # Clean start for R1 - session ID captured after completion
                await self._engine.run_agent(
                    codex_task, codex_runner, mode="exec",
                    prompt=_codex_prompt, working_directory=working_directory, enable_search=config.codex.enable_search,
                    timeout=timeout or 0,
                )
                elapsed = (datetime.now() - round1_start).total_seconds()
                status = "completed" if codex_task.status == "completed" else "failed"
                self.log(f"Codex {status} ({elapsed:.1f}s)")
                await self.notify(f"Codex {status} ({elapsed:.1f}s)")

            codex_task.async_task = asyncio.create_task(run_codex())
            async_tasks.append(codex_task.async_task)

        if "gemini" not in excluded:
            gemini_role = roles.get("gemini")
            gemini_prompt = inject_role_prefix(prompt, gemini_role)

            gemini_task = self._engine.create_task(
                command=f"council_{Agent.GEMINI.value}",
                args={"prompt": gemini_prompt, "working_directory": working_directory},
                context=self.context,
            )
            tasks["gemini"] = gemini_task

            # Capture prompt in closure
            _gemini_prompt = gemini_prompt

            async def run_gemini():
                # Clean start for R1 - session ID captured after completion
                await self._engine.run_agent(
                    gemini_task, gemini_runner, mode="exec",
                    prompt=_gemini_prompt, working_directory=working_directory,
                    timeout=timeout or 0,
                )
                elapsed = (datetime.now() - round1_start).total_seconds()
                status = "completed" if gemini_task.status == "completed" else "failed"
                self.log(f"Gemini {status} ({elapsed:.1f}s)")
                await self.notify(f"Gemini {status} ({elapsed:.1f}s)")

            gemini_task.async_task = asyncio.create_task(run_gemini())
            async_tasks.append(gemini_task.async_task)

        if "opencode" not in excluded:
            opencode_role = roles.get("opencode")
            opencode_prompt = inject_role_prefix(prompt, opencode_role)

            opencode_task = self._engine.create_task(
                command=f"council_{Agent.OPENCODE.value}",
                args={"prompt": opencode_prompt, "working_directory": working_directory},
                context=self.context,
            )
            tasks["opencode"] = opencode_task

            # Capture prompt in closure
            _opencode_prompt = opencode_prompt

            async def run_opencode():
                # Clean start for R1 - session ID captured after completion
                await self._engine.run_agent(
                    opencode_task, opencode_runner, mode="exec",
                    prompt=_opencode_prompt, working_directory=working_directory,
                    timeout=timeout or 0,
                )
                elapsed = (datetime.now() - round1_start).total_seconds()
                status = "completed" if opencode_task.status == "completed" else "failed"
                self.log(f"OpenCode {status} ({elapsed:.1f}s)")
                await self.notify(f"OpenCode {status} ({elapsed:.1f}s)")

            opencode_task.async_task = asyncio.create_task(run_opencode())
            async_tasks.append(opencode_task.async_task)

        if "claudeor" not in excluded and config.claudeor.api_key:
            claudeor_role = roles.get("claudeor")
            claudeor_prompt = inject_role_prefix(prompt, claudeor_role)

            claudeor_task = self._engine.create_task(
                command=f"council_{Agent.CLAUDEOR.value}",
                args={"prompt": claudeor_prompt, "working_directory": working_directory},
                context=self.context,
            )
            tasks["claudeor"] = claudeor_task

            # Capture prompt in closure
            _claudeor_prompt = claudeor_prompt

            async def run_claudeor():
                # Clean start for R1 - session ID captured after completion
                await self._engine.run_agent(
                    claudeor_task, claudeor_runner, mode="exec",
                    prompt=_claudeor_prompt, working_directory=working_directory,
                    timeout=timeout or 0,
                )
                elapsed = (datetime.now() - round1_start).total_seconds()
                status = "completed" if claudeor_task.status == "completed" else "failed"
                model_name = config.claudeor.model or "Claude/OpenRouter"
                self.log(f"{model_name} {status} ({elapsed:.1f}s)")
                await self.notify(f"{model_name} {status} ({elapsed:.1f}s)")

            claudeor_task.async_task = asyncio.create_task(run_claudeor())
            async_tasks.append(claudeor_task.async_task)

        if "aichat" not in excluded:
            aichat_role = roles.get("aichat")
            aichat_prompt = inject_role_prefix(prompt, aichat_role)

            aichat_task = self._engine.create_task(
                command=f"council_{Agent.AICHAT.value}",
                args={"prompt": aichat_prompt, "working_directory": working_directory},
                context=self.context,
            )
            tasks["aichat"] = aichat_task

            # Capture prompt in closure
            _aichat_prompt = aichat_prompt

            async def run_aichat():
                await self._engine.run_agent(
                    aichat_task, aichat_runner, mode="exec",
                    prompt=_aichat_prompt, working_directory=working_directory,
                    timeout=timeout or 0,
                )
                elapsed = (datetime.now() - round1_start).total_seconds()
                status = "completed" if aichat_task.status == "completed" else "failed"
                model_name = config.aichat.model or "AiChat"
                self.log(f"{model_name} {status} ({elapsed:.1f}s)")
                await self.notify(f"{model_name} {status} ({elapsed:.1f}s)")

            aichat_task.async_task = asyncio.create_task(run_aichat())
            async_tasks.append(aichat_task.async_task)

        # Wait for all tasks with timeout
        if async_tasks:
            done, pending = await asyncio.wait(
                async_tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )

            # Kill subprocesses for any tasks that timed out
            for task in tasks.values():
                if task.async_task in pending:
                    self.log(f"{task.command} timed out")
                    await self._engine.kill_task_subprocess(task)
                    task.status = "failed"
                    task.error = f"Timed out after {timeout} seconds" if timeout else "Timed out"
                    task.completion_time = datetime.now()

        round1_elapsed = (datetime.now() - round1_start).total_seconds()
        self.log(f"Round 1 complete ({round1_elapsed:.1f}s)")

        # Capture session IDs from completed R1 agents for R2 resume.
        # Each runner gets its agent's captured output; codex and claudeor read
        # the ID from it, the others use since_mtime and working_directory.
        # Subtract 1 second to handle filesystem mtime granularity (some filesystems
        # only store whole seconds, so a session created at 100.9s might have mtime 100)
        r1_start_mtime = round1_start.timestamp() - 1.0

        # Parse session IDs in parallel to reduce inter-round latency
        async def parse_codex_session():
            if "codex" not in tasks or tasks["codex"].status != "completed":
                return None
            session = await codex_runner.parse_session_id(
                _captured_output(tasks["codex"]),
                since_mtime=r1_start_mtime, working_directory=working_directory,
            )
            if session and not codex_runner.validate_session_id(session):
                self.log(f"Codex session ID validation failed: {session}")
                return None
            if not session:
                self.log("Codex session ID not found, R2 will use exec mode")
            return session

        async def parse_gemini_session():
            if "gemini" not in tasks or tasks["gemini"].status != "completed":
                return None
            session = await gemini_runner.parse_session_id(
                _captured_output(tasks["gemini"]),
                since_mtime=r1_start_mtime, working_directory=working_directory,
            )
            if session and not gemini_runner.validate_session_id(session):
                self.log(f"Gemini session ID validation failed: {session}")
                return None
            if not session:
                self.log("Gemini session ID not found, R2 will use exec mode")
            return session

        async def parse_opencode_session():
            if "opencode" not in tasks or tasks["opencode"].status != "completed":
                return None
            session = await opencode_runner.parse_session_id(
                _captured_output(tasks["opencode"]),
                since_mtime=r1_start_mtime, working_directory=working_directory,
            )
            if session and not opencode_runner.validate_session_id(session):
                self.log(f"OpenCode session ID validation failed: {session}")
                return None
            if not session:
                self.log("OpenCode session ID not found, R2 will use exec mode")
            return session

        async def parse_claudeor_session():
            if "claudeor" not in tasks or tasks["claudeor"].status != "completed":
                return None
            session = await claudeor_runner.parse_session_id(
                _captured_output(tasks["claudeor"]),
                since_mtime=r1_start_mtime, working_directory=working_directory,
            )
            if session and not claudeor_runner.validate_session_id(session):
                self.log(f"ClaudeOR session ID validation failed: {session}")
                return None
            if not session:
                self.log("ClaudeOR session ID not found, R2 will use exec mode")
            return session

        async def parse_aichat_session():
            if "aichat" not in tasks or tasks["aichat"].status != "completed":
                return None
            session = await aichat_runner.parse_session_id(
                _captured_output(tasks["aichat"]),
                since_mtime=r1_start_mtime, working_directory=working_directory,
            )
            if session and not aichat_runner.validate_session_id(session):
                self.log(f"AiChat session ID validation failed: {session}")
                return None
            if not session:
                self.log("AiChat session ID not found, R2 will use exec mode")
            return session

        codex_session, gemini_session, opencode_session, claudeor_session, aichat_session = await asyncio.gather(
            parse_codex_session(),
            parse_gemini_session(),
            parse_opencode_session(),
            parse_claudeor_session(),
            parse_aichat_session(),
        )

        return CouncilRound(
            codex=build_agent_response(tasks["codex"], Agent.CODEX, session_id=codex_session) if "codex" in tasks else None,
            gemini=build_agent_response(tasks["gemini"], Agent.GEMINI, session_id=gemini_session) if "gemini" in tasks else None,
            opencode=build_agent_response(tasks["opencode"], Agent.OPENCODE, session_id=opencode_session) if "opencode" in tasks else None,
            claudeor=build_agent_response(tasks["claudeor"], Agent.CLAUDEOR, session_id=claudeor_session) if "claudeor" in tasks else None,
            aichat=build_agent_response(tasks["aichat"], Agent.AICHAT, session_id=aichat_session) if "aichat" in tasks else None,
        )

    async def _run_round_2(
        self,
        prompt: str,
        working_directory: str | None,
        round_1: CouncilRound,
        claude_opinion: str | None,
        critique: bool,
        timeout: int | None,
        roles: dict[str, RoleDefinition],
    ) -> CouncilRound:
        """Run the second round of deliberation.

        Option A: R2 uses explicit session IDs from R1 to resume.
        Falls back to exec mode if session ID is not available.
        Role prefixes are injected (sticky roles) to maintain perspective.
        """
        self.log("Round 2: deliberation phase...")
        excluded = config.council.exclude_agents

        # Skip agents that failed/timed out in R1 - no point retrying
        r1_failed = set()
        for agent_name, r1_result in [
            ("codex", round_1.codex),
            ("gemini", round_1.gemini),
            ("opencode", round_1.opencode),
            ("claudeor", round_1.claudeor),
            ("aichat", round_1.aichat),
        ]:
            if r1_result and r1_result.status == "failed":
                r1_failed.add(agent_name)
                self.log(f"Skipping {agent_name} in R2 (failed in R1)")

        # Get explicit session IDs from R1 (Option A)
        # If session_id is None, we fall back to exec mode
        codex_session = round_1.codex.session_id if round_1.codex else None
        gemini_session = round_1.gemini.session_id if round_1.gemini else None
        opencode_session = round_1.opencode.session_id if round_1.opencode else None
        claudeor_session = round_1.claudeor.session_id if round_1.claudeor else None
        aichat_session = round_1.aichat.session_id if round_1.aichat else None

        codex_content = (round_1.codex.content or round_1.codex.error or "(no response)") if round_1.codex else None
        gemini_content = (round_1.gemini.content or round_1.gemini.error or "(no response)") if round_1.gemini else None
        opencode_content = (round_1.opencode.content or round_1.opencode.error or "(no response)") if round_1.opencode else None
        claudeor_content = (round_1.claudeor.content or round_1.claudeor.error or "(no response)") if round_1.claudeor else None
        aichat_content = (round_1.aichat.content or round_1.aichat.error or "(no response)") if round_1.aichat else None
        claude_content = claude_opinion.strip() if claude_opinion else None

        round2_start = datetime.now()
        tasks = {}
        async_tasks = []

        # R2 resumes from R1 sessions using explicit session IDs (Option A)
        # Codex: uses resume with explicit session ID from R1
        # Gemini: uses resume with explicit session index from R1
        # OpenCode: uses resume with explicit session ID from R1
        #
        # If session ID is None (R1 failed or parsing failed), fall back to exec mode
        # Role prefixes (sticky roles) are injected to maintain perspective

        if "codex" not in excluded and "codex" not in r1_failed:
            codex_role = roles.get("codex")
            # Build prompt for resume mode (no original needed - agent has R1 context)
            codex_delib_prompt_resume = build_deliberation_prompt_with_role(
                original_prompt=prompt,
                role=codex_role,
                codex_answer=codex_content,
                gemini_answer=gemini_content,
                opencode_answer=opencode_content,
                claudeor_answer=claudeor_content,
                aichat_answer=aichat_content,
                claude_answer=claude_content,
                critique=critique,
                include_original=False,
            )
            # Build prompt for exec fallback (include original - agent starts fresh)
            codex_delib_prompt_exec = build_deliberation_prompt_with_role(
                original_prompt=prompt,
                role=codex_role,
                codex_answer=codex_content,
                gemini_answer=gemini_content,
                opencode_answer=opencode_content,
                claudeor_answer=claudeor_content,
                aichat_answer=aichat_content,
                claude_answer=claude_content,
                critique=critique,
                include_original=True,
            )

            codex_delib_task = self._engine.create_task(
                command=f"council_{Agent.CODEX.value}_delib",
                args={"prompt": codex_delib_prompt_resume, "working_directory": working_directory},
                context=self.context,
            )
            tasks["codex"] = codex_delib_task

            # Capture session and prompts in closure
            _codex_session = codex_session
            _codex_delib_prompt_resume = codex_delib_prompt_resume
            _codex_delib_prompt_exec = codex_delib_prompt_exec

            async def run_codex_delib():
                # Resume with explicit session ID if available, otherwise exec with full context
                if _codex_session:
                    await self._engine.run_agent(
                        codex_delib_task, codex_runner, mode="resume",
                        session_ref=_codex_session,
                        prompt=_codex_delib_prompt_resume, working_directory=working_directory,
                        enable_search=config.codex.enable_search,
                        timeout=timeout or 0,
                    )
                else:
                    await self._engine.run_agent(
                        codex_delib_task, codex_runner, mode="exec",
                        prompt=_codex_delib_prompt_exec, working_directory=working_directory,
                        enable_search=config.codex.enable_search,
                        timeout=timeout or 0,
                    )
                elapsed = (datetime.now() - round2_start).total_seconds()
                self.log(f"Codex revised ({elapsed:.1f}s)")
                await self.notify(f"Codex revised ({elapsed:.1f}s)")

            codex_delib_task.async_task = asyncio.create_task(run_codex_delib())
            async_tasks.append(codex_delib_task.async_task)

        if "gemini" not in excluded and "gemini" not in r1_failed:
            gemini_role = roles.get("gemini")
            # Build prompt for resume mode (no original needed - agent has R1 context)
            gemini_delib_prompt_resume = build_deliberation_prompt_with_role(
                original_prompt=prompt,
                role=gemini_role,
                codex_answer=codex_content,
                gemini_answer=gemini_content,
                opencode_answer=opencode_content,
                claudeor_answer=claudeor_content,
                aichat_answer=aichat_content,
                claude_answer=claude_content,
                critique=critique,
                include_original=False,
            )
            # Build prompt for exec fallback (include original - agent starts fresh)
            gemini_delib_prompt_exec = build_deliberation_prompt_with_role(
                original_prompt=prompt,
                role=gemini_role,
                codex_answer=codex_content,
                gemini_answer=gemini_content,
                opencode_answer=opencode_content,
                claudeor_answer=claudeor_content,
                aichat_answer=aichat_content,
                claude_answer=claude_content,
                critique=critique,
                include_original=True,
            )

            gemini_delib_task = self._engine.create_task(
                command=f"council_{Agent.GEMINI.value}_delib",
                args={"prompt": gemini_delib_prompt_resume, "working_directory": working_directory},
                context=self.context,
            )
            tasks["gemini"] = gemini_delib_task

            # Capture session and prompts in closure
            _gemini_session = gemini_session
            _gemini_delib_prompt_resume = gemini_delib_prompt_resume
            _gemini_delib_prompt_exec = gemini_delib_prompt_exec

            async def run_gemini_delib():
                # Resume with explicit session index if available, otherwise exec with full context
                if _gemini_session:
                    await self._engine.run_agent(
                        gemini_delib_task, gemini_runner, mode="resume",
                        session_ref=_gemini_session,
                        prompt=_gemini_delib_prompt_resume, working_directory=working_directory,
                        timeout=timeout or 0,
                    )
                else:
                    await self._engine.run_agent(
                        gemini_delib_task, gemini_runner, mode="exec",
                        prompt=_gemini_delib_prompt_exec, working_directory=working_directory,
                        timeout=timeout or 0,
                    )
                elapsed = (datetime.now() - round2_start).total_seconds()
                self.log(f"Gemini revised ({elapsed:.1f}s)")
                await self.notify(f"Gemini revised ({elapsed:.1f}s)")

            gemini_delib_task.async_task = asyncio.create_task(run_gemini_delib())
            async_tasks.append(gemini_delib_task.async_task)

        if "opencode" not in excluded and "opencode" not in r1_failed:
            opencode_role = roles.get("opencode")
            # Build prompt for resume mode (no original needed - agent has R1 context)
            opencode_delib_prompt_resume = build_deliberation_prompt_with_role(
                original_prompt=prompt,
                role=opencode_role,
                codex_answer=codex_content,
                gemini_answer=gemini_content,
                opencode_answer=opencode_content,
                claudeor_answer=claudeor_content,
                aichat_answer=aichat_content,
                claude_answer=claude_content,
                critique=critique,
                include_original=False,
            )
            # Build prompt for exec fallback (include original - agent starts fresh)
            opencode_delib_prompt_exec = build_deliberation_prompt_with_role(
                original_prompt=prompt,
                role=opencode_role,
                codex_answer=codex_content,
                gemini_answer=gemini_content,
                opencode_answer=opencode_content,
                claudeor_answer=claudeor_content,
                aichat_answer=aichat_content,
                claude_answer=claude_content,
                critique=critique,
                include_original=True,
            )

            opencode_delib_task = self._engine.create_task(
                command=f"council_{Agent.OPENCODE.value}_delib",
                args={"prompt": opencode_delib_prompt_resume, "working_directory": working_directory},
                context=self.context,
            )
            tasks["opencode"] = opencode_delib_task

            # Capture session and prompts in closure
            _opencode_session = opencode_session
            _opencode_delib_prompt_resume = opencode_delib_prompt_resume
            _opencode_delib_prompt_exec = opencode_delib_prompt_exec

            async def run_opencode_delib():
                # Resume with explicit session ID if available, otherwise exec with full context
                if _opencode_session:
                    await self._engine.run_agent(
                        opencode_delib_task, opencode_runner, mode="resume",
                        session_ref=_opencode_session,
                        prompt=_opencode_delib_prompt_resume, working_directory=working_directory,
                        timeout=timeout or 0,
                    )
                else:
                    await self._engine.run_agent(
                        opencode_delib_task, opencode_runner, mode="exec",
                        prompt=_opencode_delib_prompt_exec, working_directory=working_directory,
                        timeout=timeout or 0,
                    )
                elapsed = (datetime.now() - round2_start).total_seconds()
                self.log(f"OpenCode revised ({elapsed:.1f}s)")
                await self.notify(f"OpenCode revised ({elapsed:.1f}s)")

            opencode_delib_task.async_task = asyncio.create_task(run_opencode_delib())
            async_tasks.append(opencode_delib_task.async_task)

        if "claudeor" not in excluded and config.claudeor.api_key and "claudeor" not in r1_failed:
            claudeor_role = roles.get("claudeor")
            # Build prompt for resume mode (no original needed - agent has R1 context)
            claudeor_delib_prompt_resume = build_deliberation_prompt_with_role(
                original_prompt=prompt,
                role=claudeor_role,
                codex_answer=codex_content,
                gemini_answer=gemini_content,
                opencode_answer=opencode_content,
                claudeor_answer=claudeor_content,
                aichat_answer=aichat_content,
                claude_answer=claude_content,
                critique=critique,
                include_original=False,
            )
            # Build prompt for exec fallback (include original - agent starts fresh)
            claudeor_delib_prompt_exec = build_deliberation_prompt_with_role(
                original_prompt=prompt,
                role=claudeor_role,
                codex_answer=codex_content,
                gemini_answer=gemini_content,
                opencode_answer=opencode_content,
                claudeor_answer=claudeor_content,
                aichat_answer=aichat_content,
                claude_answer=claude_content,
                critique=critique,
                include_original=True,
            )

            claudeor_delib_task = self._engine.create_task(
                command=f"council_{Agent.CLAUDEOR.value}_delib",
                args={"prompt": claudeor_delib_prompt_resume, "working_directory": working_directory},
                context=self.context,
            )
            tasks["claudeor"] = claudeor_delib_task

            # Capture session and prompts in closure
            _claudeor_session = claudeor_session
            _claudeor_delib_prompt_resume = claudeor_delib_prompt_resume
            _claudeor_delib_prompt_exec = claudeor_delib_prompt_exec

            async def run_claudeor_delib():
                # Resume with explicit session ID if available, otherwise exec with full context
                if _claudeor_session:
                    await self._engine.run_agent(
                        claudeor_delib_task, claudeor_runner, mode="resume",
                        session_ref=_claudeor_session,
                        prompt=_claudeor_delib_prompt_resume, working_directory=working_directory,
                        timeout=timeout or 0,
                    )
                else:
                    await self._engine.run_agent(
                        claudeor_delib_task, claudeor_runner, mode="exec",
                        prompt=_claudeor_delib_prompt_exec, working_directory=working_directory,
                        timeout=timeout or 0,
                    )
                elapsed = (datetime.now() - round2_start).total_seconds()
                model_name = config.claudeor.model or "Claude/OpenRouter"
                self.log(f"{model_name} revised ({elapsed:.1f}s)")
                await self.notify(f"{model_name} revised ({elapsed:.1f}s)")

            claudeor_delib_task.async_task = asyncio.create_task(run_claudeor_delib())
            async_tasks.append(claudeor_delib_task.async_task)

        if "aichat" not in excluded and "aichat" not in r1_failed:
            aichat_role = roles.get("aichat")
            aichat_delib_prompt_resume = build_deliberation_prompt_with_role(
                original_prompt=prompt,
                role=aichat_role,
                codex_answer=codex_content,
                gemini_answer=gemini_content,
                opencode_answer=opencode_content,
                claudeor_answer=claudeor_content,
                aichat_answer=aichat_content,
                claude_answer=claude_content,
                critique=critique,
                include_original=False,
            )
            aichat_delib_prompt_exec = build_deliberation_prompt_with_role(
                original_prompt=prompt,
                role=aichat_role,
                codex_answer=codex_content,
                gemini_answer=gemini_content,
                opencode_answer=opencode_content,
                claudeor_answer=claudeor_content,
                aichat_answer=aichat_content,
                claude_answer=claude_content,
                critique=critique,
                include_original=True,
            )

            aichat_delib_task = self._engine.create_task(
                command=f"council_{Agent.AICHAT.value}_delib",
                args={"prompt": aichat_delib_prompt_resume, "working_directory": working_directory},
                context=self.context,
            )
            tasks["aichat"] = aichat_delib_task

            _aichat_session = aichat_session
            _aichat_delib_prompt_resume = aichat_delib_prompt_resume
            _aichat_delib_prompt_exec = aichat_delib_prompt_exec

            async def run_aichat_delib():
                if _aichat_session:
                    await self._engine.run_agent(
                        aichat_delib_task, aichat_runner, mode="resume",
                        session_ref=_aichat_session,
                        prompt=_aichat_delib_prompt_resume, working_directory=working_directory,
                        timeout=timeout or 0,
                    )
                else:
                    await self._engine.run_agent(
                        aichat_delib_task, aichat_runner, mode="exec",
                        prompt=_aichat_delib_prompt_exec, working_directory=working_directory,
                        timeout=timeout or 0,
                    )
                elapsed = (datetime.now() - round2_start).total_seconds()
                model_name = config.aichat.model or "AiChat"
                self.log(f"{model_name} revised ({elapsed:.1f}s)")
                await self.notify(f"{model_name} revised ({elapsed:.1f}s)")

            aichat_delib_task.async_task = asyncio.create_task(run_aichat_delib())
            async_tasks.append(aichat_delib_task.async_task)

        # Wait for all tasks with timeout
        if async_tasks:
            done, pending = await asyncio.wait(
                async_tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )

            # Kill subprocesses for any tasks that timed out
            for task in tasks.values():
                if task.async_task in pending:
                    self.log(f"{task.command} timed out")
                    await self._engine.kill_task_subprocess(task)
                    task.status = "failed"
                    task.error = f"Timed out after {timeout} seconds" if timeout else "Timed out"
                    task.completion_time = datetime.now()

        round2_elapsed = (datetime.now() - round2_start).total_seconds()
        self.log(f"Round 2 complete ({round2_elapsed:.1f}s)")
        await self.notify("Council Round 2 complete", progress=90)

        return CouncilRound(
            codex=build_agent_response(tasks["codex"], Agent.CODEX) if "codex" in tasks else None,
            gemini=build_agent_response(tasks["gemini"], Agent.GEMINI) if "gemini" in tasks else None,
            opencode=build_agent_response(tasks["opencode"], Agent.OPENCODE) if "opencode" in tasks else None,
            claudeor=build_agent_response(tasks["claudeor"], Agent.CLAUDEOR) if "claudeor" in tasks else None,
            aichat=build_agent_response(tasks["aichat"], Agent.AICHAT) if "aichat" in tasks else None,
        )
