"""
Tests for session management logic.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from owlex.agents.codex import CodexRunner
from owlex.agents.claudeor import ClaudeORRunner
from owlex.agents.gemini import GeminiRunner, get_gemini_session_for_project, _find_gemini_project_dir
from owlex.agents.opencode import OpenCodeRunner, get_latest_opencode_session, _get_opencode_project_id
from owlex.agents.base import AgentRunner


class TestSessionIdValidation:
    """Tests for session ID validation in base runner."""

    def setup_method(self):
        self.codex = CodexRunner()
        self.gemini = GeminiRunner()
        self.opencode = OpenCodeRunner()

    def test_rejects_empty_session_id(self):
        assert not self.codex.validate_session_id("")
        assert not self.gemini.validate_session_id("")
        assert not self.opencode.validate_session_id("")

    def test_rejects_flag_injection(self):
        """Session IDs starting with dash could be interpreted as CLI flags."""
        assert not self.codex.validate_session_id("--help")
        assert not self.codex.validate_session_id("-v")
        assert not self.gemini.validate_session_id("--version")
        assert not self.opencode.validate_session_id("-n")

    def test_rejects_shell_metacharacters(self):
        """Session IDs with shell metacharacters could enable command injection."""
        dangerous_chars = [";", "|", "&", "$", "`", "(", ")", "{", "}", "<", ">", "\n", "\r"]
        for char in dangerous_chars:
            bad_id = f"session{char}evil"
            assert not self.codex.validate_session_id(bad_id), f"Should reject '{char}'"
            assert not self.opencode.validate_session_id(bad_id), f"Should reject '{char}'"

    def test_accepts_valid_uuid_style_session(self):
        """Valid UUID-style session IDs should be accepted."""
        valid_uuid = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert self.codex.validate_session_id(valid_uuid)
        assert self.opencode.validate_session_id(valid_uuid)

    def test_accepts_valid_opencode_session_id(self):
        """OpenCode uses ses_XXX format."""
        valid_id = "ses_49b5d1b81ffeZfa2uTg3NVmKrH"
        assert self.opencode.validate_session_id(valid_id)


class TestGeminiSessionValidation:
    """Tests for Gemini-specific session validation."""

    def setup_method(self):
        self.gemini = GeminiRunner()

    def test_accepts_numeric_index(self):
        """Gemini uses 1-indexed numeric indices for -r flag."""
        assert self.gemini.validate_session_id("1")
        assert self.gemini.validate_session_id("5")
        assert self.gemini.validate_session_id("100")

    def test_rejects_zero_index(self):
        """Gemini uses 1-indexed sessions, so 0 is invalid."""
        assert not self.gemini.validate_session_id("0")

    def test_accepts_latest_keyword(self):
        """Gemini accepts 'latest' as a session reference."""
        assert self.gemini.validate_session_id("latest")

    def test_rejects_non_numeric(self):
        """Gemini only accepts numeric indices or 'latest'."""
        assert not self.gemini.validate_session_id("abc")
        assert not self.gemini.validate_session_id("session-1")

    def test_rejects_flag_injection(self):
        """Even Gemini-specific validation rejects flags."""
        assert not self.gemini.validate_session_id("--help")
        assert not self.gemini.validate_session_id("-v")


class TestProjectDirectoryDiscovery:
    """Tests for project directory discovery matching CLI behavior."""

    def test_gemini_finds_project_dir_by_project_root(self):
        """Gemini should find project dir by reading .project_root files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / ".gemini" / "tmp" / "myproject"
            project_dir.mkdir(parents=True)
            (project_dir / ".project_root").write_text("/test/project")

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _find_gemini_project_dir("/test/project")
                assert result == project_dir

    def test_gemini_returns_none_for_unknown_project(self):
        """Should return None when no .project_root matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / ".gemini" / "tmp" / "other"
            project_dir.mkdir(parents=True)
            (project_dir / ".project_root").write_text("/other/project")

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _find_gemini_project_dir("/test/project")
                assert result is None

    def test_gemini_skips_dirs_without_project_root(self):
        """Should skip directories that don't have a .project_root file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Dir without .project_root (like 'bin')
            bin_dir = Path(tmpdir) / ".gemini" / "tmp" / "bin"
            bin_dir.mkdir(parents=True)

            # Proper project dir
            project_dir = Path(tmpdir) / ".gemini" / "tmp" / "myproject"
            project_dir.mkdir(parents=True)
            (project_dir / ".project_root").write_text("/test/project")

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _find_gemini_project_dir("/test/project")
                assert result == project_dir

    def test_gemini_normalizes_paths(self):
        """Path comparison should handle trailing slashes and normalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / ".gemini" / "tmp" / "myproject"
            project_dir.mkdir(parents=True)
            (project_dir / ".project_root").write_text("/test/project")

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                # Trailing slash
                result = _find_gemini_project_dir("/test/project/")
                assert result == project_dir

                # Double slash
                result = _find_gemini_project_dir("/test//project")
                assert result == project_dir

    def test_gemini_handles_newline_in_project_root(self):
        """Should handle .project_root files with trailing newlines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / ".gemini" / "tmp" / "myproject"
            project_dir.mkdir(parents=True)
            (project_dir / ".project_root").write_text("/test/project\n")

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _find_gemini_project_dir("/test/project")
                assert result == project_dir

    def test_gemini_returns_none_when_gemini_dir_missing(self):
        """Should return None when ~/.gemini/tmp doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _find_gemini_project_dir("/test/project")
                assert result is None

    def test_gemini_handles_corrupt_project_root(self):
        """Should skip .project_root files with invalid bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dir with corrupt .project_root
            corrupt_dir = Path(tmpdir) / ".gemini" / "tmp" / "corrupt"
            corrupt_dir.mkdir(parents=True)
            (corrupt_dir / ".project_root").write_bytes(b"\x80\x81\x82\xff")

            # Create valid project dir
            project_dir = Path(tmpdir) / ".gemini" / "tmp" / "myproject"
            project_dir.mkdir(parents=True)
            (project_dir / ".project_root").write_text("/test/project")

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _find_gemini_project_dir("/test/project")
                assert result == project_dir

    def test_gemini_handles_relative_input_path(self):
        """Should resolve relative paths before comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / ".gemini" / "tmp" / "myproject"
            project_dir.mkdir(parents=True)
            cwd = os.getcwd()
            (project_dir / ".project_root").write_text(cwd)

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _find_gemini_project_dir(".")
                assert result == project_dir

    def test_opencode_project_id_lookup(self):
        """OpenCode projectID should be looked up from config files, not computed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "project")
            os.makedirs(test_path)

            project_dir = Path(tmpdir) / ".local" / "share" / "opencode" / "storage" / "project"
            project_dir.mkdir(parents=True)
            import json
            project_id = "mock_project_id_12345"
            project_file = project_dir / f"{project_id}.json"
            with open(project_file, "w") as f:
                json.dump({"id": project_id, "worktree": os.path.abspath(test_path)}, f)

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _get_opencode_project_id(test_path)
                assert result == project_id

    def test_opencode_project_id_returns_none_for_unknown(self):
        """OpenCode projectID lookup should return None for unknown projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / ".local" / "share" / "opencode" / "storage" / "project"
            project_dir.mkdir(parents=True)

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _get_opencode_project_id("/some/unknown/path")
                assert result is None

    def test_opencode_handles_trailing_slash(self):
        """OpenCode should match projects with trailing slash differences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "project")
            os.makedirs(test_path)

            project_dir = Path(tmpdir) / ".local" / "share" / "opencode" / "storage" / "project"
            project_dir.mkdir(parents=True)
            import json
            project_id = "mock_project_id_12345"
            project_file = project_dir / f"{project_id}.json"
            with open(project_file, "w") as f:
                json.dump({"id": project_id, "worktree": test_path + "/"}, f)

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _get_opencode_project_id(test_path)
                assert result == project_id

    def test_opencode_handles_null_worktree(self):
        """OpenCode should skip project configs with null/missing worktree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / ".local" / "share" / "opencode" / "storage" / "project"
            project_dir.mkdir(parents=True)
            import json

            # Project with null worktree
            null_file = project_dir / "null_project.json"
            with open(null_file, "w") as f:
                json.dump({"id": "null_project", "worktree": None}, f)

            # Project with missing worktree
            missing_file = project_dir / "missing_project.json"
            with open(missing_file, "w") as f:
                json.dump({"id": "missing_project"}, f)

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _get_opencode_project_id("/test/project")
                assert result is None

    def test_opencode_handles_expanduser_worktree(self):
        """OpenCode should expand ~ in stored worktree paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            home = str(Path.home())

            project_dir = Path(tmpdir) / ".local" / "share" / "opencode" / "storage" / "project"
            project_dir.mkdir(parents=True)
            import json
            project_id = "expanduser_project"
            project_file = project_dir / f"{project_id}.json"
            with open(project_file, "w") as f:
                json.dump({"id": project_id, "worktree": "~/repos/testproject"}, f)

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _get_opencode_project_id(os.path.join(home, "repos", "testproject"))
                assert result == project_id

    def test_opencode_skips_non_string_project_id(self):
        """OpenCode should skip project configs with non-string id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / ".local" / "share" / "opencode" / "storage" / "project"
            project_dir.mkdir(parents=True)
            import json

            # Project with numeric id
            bad_file = project_dir / "bad_project.json"
            with open(bad_file, "w") as f:
                json.dump({"id": 12345, "worktree": "/test/project"}, f)

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _get_opencode_project_id("/test/project")
                assert result is None

    def test_opencode_handles_relative_input_path(self):
        """OpenCode should resolve relative paths before comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()

            project_dir = Path(tmpdir) / ".local" / "share" / "opencode" / "storage" / "project"
            project_dir.mkdir(parents=True)
            import json
            project_id = "mock_project_id_12345"
            project_file = project_dir / f"{project_id}.json"
            with open(project_file, "w") as f:
                json.dump({"id": project_id, "worktree": cwd}, f)

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = _get_opencode_project_id(".")
                assert result == project_id


class TestAsyncSessionParsing:
    """Tests for async session parsing methods."""

    @pytest.mark.asyncio
    async def test_codex_parse_session_id_is_async(self):
        """Codex parse_session_id should be awaitable (coroutine)."""
        runner = CodexRunner()
        # Verify it's a coroutine function
        import inspect
        assert inspect.iscoroutinefunction(runner.parse_session_id)
        # Should be awaitable without raising
        result = await runner.parse_session_id("")
        # Result can be None or a session ID depending on system state
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_gemini_parse_session_id_is_async(self):
        """Gemini parse_session_id should be awaitable (coroutine)."""
        runner = GeminiRunner()
        import inspect
        assert inspect.iscoroutinefunction(runner.parse_session_id)
        result = await runner.parse_session_id("")
        # Result can be None or "1" depending on system state
        assert result is None or result == "1"

    @pytest.mark.asyncio
    async def test_opencode_parse_session_id_is_async(self):
        """OpenCode parse_session_id should be awaitable (coroutine)."""
        runner = OpenCodeRunner()
        import inspect
        assert inspect.iscoroutinefunction(runner.parse_session_id)
        result = await runner.parse_session_id("")
        # Result can be None or a session ID depending on system state
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_base_parse_session_id_is_async(self):
        """Base AgentRunner parse_session_id should be awaitable."""
        # Create a concrete subclass for testing
        class TestRunner(AgentRunner):
            @property
            def name(self):
                return "test"

            def build_exec_command(self, prompt, **kwargs):
                pass

            def build_resume_command(self, session_ref, prompt, **kwargs):
                pass

            def get_output_cleaner(self):
                return lambda x, y: x

        runner = TestRunner()
        import inspect
        assert inspect.iscoroutinefunction(runner.parse_session_id)
        result = await runner.parse_session_id("")
        assert result is None  # Base class returns None


class TestSessionIdFromOutput:
    """Codex and ClaudeOR read the session ID from their own output, never the filesystem."""

    # stderr header of codex-cli 0.159.1, followed by the answer on stdout
    CODEX_OUTPUT = (
        "pong\n"
        "OpenAI Codex v0.159.1\n"
        "--------\n"
        "workdir: /tmp/project\n"
        "model: gpt-6.1-sol\n"
        "provider: openai\n"
        "approval: never\n"
        "sandbox: workspace-write\n"
        "session id: 01a108dd-412d-7d42-86df-fce173f73055\n"
        "--------\n"
        "user\n"
        "Reply with exactly: pong\n"
    )

    # `claude -p --output-format json` from Claude Code 2.1.289 (trimmed)
    CLAUDE_OUTPUT = (
        '{"type":"result","subtype":"success","is_error":false,'
        '"result":"ok","stop_reason":"end_turn",'
        '"session_id":"2cf8b2f3-0eb0-49ff-ade5-19d714991538","total_cost_usd":0.14}'
    )

    async def test_codex_reads_session_id_line(self):
        runner = CodexRunner()
        assert await runner.parse_session_id(self.CODEX_OUTPUT) == "01a108dd-412d-7d42-86df-fce173f73055"

    async def test_codex_handles_crlf(self):
        runner = CodexRunner()
        output = self.CODEX_OUTPUT.replace("\n", "\r\n")
        assert await runner.parse_session_id(output) == "01a108dd-412d-7d42-86df-fce173f73055"

    async def test_codex_returns_none_without_header(self):
        runner = CodexRunner()
        assert await runner.parse_session_id("pong\n") is None
        assert await runner.parse_session_id("") is None

    async def test_codex_ignores_session_files(self):
        """A newer session file on disk must not be picked up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            day = Path(tmpdir) / ".codex" / "sessions" / "2026" / "10" / "04"
            day.mkdir(parents=True)
            (day / "rollout-2026-10-04T12-00-00-11111111-2222-3333-4444-555555555555.jsonl").touch()

            runner = CodexRunner()
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                assert await runner.parse_session_id("pong\n", since_mtime=0) is None

    async def test_claudeor_reads_json_session_id(self):
        runner = ClaudeORRunner()
        assert await runner.parse_session_id(self.CLAUDE_OUTPUT) == "2cf8b2f3-0eb0-49ff-ade5-19d714991538"

    async def test_claudeor_reads_json_followed_by_stderr(self):
        """The council appends stderr after stdout."""
        runner = ClaudeORRunner()
        output = self.CLAUDE_OUTPUT + "\nsome warning on stderr"
        assert await runner.parse_session_id(output) == "2cf8b2f3-0eb0-49ff-ade5-19d714991538"

    async def test_claudeor_returns_none_for_text_or_empty(self):
        runner = ClaudeORRunner()
        assert await runner.parse_session_id("plain text answer") is None
        assert await runner.parse_session_id("") is None
        assert await runner.parse_session_id('{"result": "ok"}') is None

    async def test_claudeor_ignores_transcripts(self):
        """A newer transcript under ~/.claude/projects must not be picked up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / ".claude" / "projects" / "D--some-project"
            project.mkdir(parents=True)
            (project / "99999999-8888-7777-6666-555555555555.jsonl").touch()

            runner = ClaudeORRunner()
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                assert await runner.parse_session_id("plain text", since_mtime=0) is None


class TestGeminiSessionDiscovery:
    """Tests for Gemini session discovery."""

    @pytest.mark.asyncio
    async def test_returns_false_when_no_gemini_dir(self):
        """Should return False when ~/.gemini/tmp doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = await get_gemini_session_for_project("/some/path")
                assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_no_working_directory(self):
        """Should return False when working_directory is None to prevent cross-project contamination."""
        result = await get_gemini_session_for_project(working_directory=None)
        assert result is False

    @pytest.mark.asyncio
    async def test_scoped_by_project_root(self):
        """Session search should be scoped by .project_root file matching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project dir with .project_root
            gemini_dir = Path(tmpdir) / ".gemini" / "tmp" / "testproject"
            chats_dir = gemini_dir / "chats"
            chats_dir.mkdir(parents=True)
            (gemini_dir / ".project_root").write_text("/test/project")

            # Create a session file
            session_file = chats_dir / "session-abc123.json"
            session_file.touch()

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                # Should find session for correct project
                result = await get_gemini_session_for_project("/test/project")
                assert result is True

                # Should not find session for different project
                result = await get_gemini_session_for_project("/other/project")
                assert result is False

    @pytest.mark.asyncio
    async def test_respects_since_mtime_filter(self):
        """Should ignore sessions older than since_mtime."""
        import time
        with tempfile.TemporaryDirectory() as tmpdir:
            gemini_dir = Path(tmpdir) / ".gemini" / "tmp" / "testproject"
            chats_dir = gemini_dir / "chats"
            chats_dir.mkdir(parents=True)
            (gemini_dir / ".project_root").write_text("/test/project")

            # Create an old session file
            session_file = chats_dir / "session-old.json"
            session_file.touch()
            old_mtime = time.time() - 3600
            os.utime(session_file, (old_mtime, old_mtime))

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                # Should be filtered out by since_mtime
                result = await get_gemini_session_for_project(
                    "/test/project", since_mtime=time.time()
                )
                assert result is False


class TestOpenCodeSessionDiscovery:
    """Tests for OpenCode session discovery."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_opencode_dir(self):
        """Should return None when storage dir doesn't exist."""
        with patch.object(Path, 'exists', return_value=False):
            result = await get_latest_opencode_session(working_directory="/some/path")
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_working_directory(self):
        """Should return None when working_directory is None to prevent cross-project contamination."""
        result = await get_latest_opencode_session(working_directory=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_extracts_session_id_from_filename(self):
        """Should extract session ID from ses_*.json filename."""
        import json as json_module
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create project config file (required for lookup)
            project_config_dir = Path(tmpdir) / ".local" / "share" / "opencode" / "storage" / "project"
            project_config_dir.mkdir(parents=True)
            project_id = "test_project_abc123"
            project_file = project_config_dir / f"{project_id}.json"
            with open(project_file, "w") as f:
                json_module.dump({"id": project_id, "worktree": "/test/project"}, f)

            # Create session directory and file
            opencode_dir = Path(tmpdir) / ".local" / "share" / "opencode" / "storage" / "session"
            project_dir = opencode_dir / project_id
            project_dir.mkdir(parents=True)

            # Create a session file
            session_file = project_dir / "ses_abc123xyz.json"
            session_file.touch()

            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = await get_latest_opencode_session(working_directory="/test/project")
                assert result == "ses_abc123xyz"


class TestServerValidation:
    """Tests for session validation in server API endpoints."""

    def test_codex_runner_validates_session_ref(self):
        """Codex runner should validate session IDs before use."""
        runner = CodexRunner()

        # Valid sessions should pass
        assert runner.validate_session_id("abc123def456")

        # Invalid sessions should fail
        assert not runner.validate_session_id("--help")
        assert not runner.validate_session_id("id;rm -rf /")

    def test_gemini_runner_validates_session_ref(self):
        """Gemini runner should validate session references."""
        runner = GeminiRunner()

        # Valid references
        assert runner.validate_session_id("1")
        assert runner.validate_session_id("latest")

        # Invalid references
        assert not runner.validate_session_id("--help")
        assert not runner.validate_session_id("abc")


class TestGeminiParseSessionIdEndToEnd:
    """End-to-end test for GeminiRunner.parse_session_id()."""

    @pytest.mark.asyncio
    async def test_parse_session_id_returns_1_with_valid_session(self):
        """parse_session_id should return '1' when a matching session exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gemini_dir = Path(tmpdir) / ".gemini" / "tmp" / "testproject"
            chats_dir = gemini_dir / "chats"
            chats_dir.mkdir(parents=True)
            (gemini_dir / ".project_root").write_text("/test/project")

            session_file = chats_dir / "session-abc123.json"
            session_file.touch()

            runner = GeminiRunner()
            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = await runner.parse_session_id(
                    "", working_directory="/test/project"
                )
                assert result == "1"

    @pytest.mark.asyncio
    async def test_parse_session_id_returns_none_for_no_match(self):
        """parse_session_id should return None when no matching project exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty gemini tmp dir
            (Path(tmpdir) / ".gemini" / "tmp").mkdir(parents=True)

            runner = GeminiRunner()
            with patch.object(Path, 'home', return_value=Path(tmpdir)):
                result = await runner.parse_session_id(
                    "", working_directory="/nonexistent/project"
                )
                assert result is None
