"""
Tests for agent CLI command construction.
"""

from unittest.mock import patch

import pytest
from owlex.agents.codex import CodexRunner
from owlex.agents.gemini import GeminiRunner
from owlex.agents.opencode import OpenCodeRunner
from owlex.agents.aichat import AiChatRunner
from owlex.agents.claudeor import ClaudeORRunner, clean_claudeor_output


class TestCodexRunner:
    """Tests for Codex CLI command construction."""

    @pytest.fixture
    def runner(self):
        return CodexRunner()

    def test_exec_basic_command(self, runner):
        """Should build basic exec command."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert cmd.command[0] == "codex"
            assert cmd.command[1] == "exec"
            assert "--skip-git-repo-check" in cmd.command
            assert "--full-auto" in cmd.command
            assert "-" in cmd.command  # stdin marker
            assert cmd.prompt == "Hello"
            assert cmd.output_prefix == "Codex Output"
            assert cmd.stream is True

    def test_exec_with_working_directory(self, runner):
        """Should add --cd flag for working directory."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_exec_command(
                prompt="Hello",
                working_directory="/path/to/dir"
            )

            assert "--cd" in cmd.command
            idx = cmd.command.index("--cd")
            assert cmd.command[idx + 1] == "/path/to/dir"

    def test_exec_with_search_enabled(self, runner):
        """Should add --enable web_search_request when search enabled."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_exec_command(
                prompt="Hello",
                enable_search=True
            )

            assert "--enable" in cmd.command
            idx = cmd.command.index("--enable")
            assert cmd.command[idx + 1] == "web_search_request"

    def test_exec_without_search(self, runner):
        """Should not include search flag when disabled."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_exec_command(
                prompt="Hello",
                enable_search=False
            )

            assert "--enable" not in cmd.command

    def test_exec_bypass_approvals(self, runner):
        """Should use bypass flag when config enables it."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = True

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--dangerously-bypass-approvals-and-sandbox" in cmd.command
            assert "--full-auto" not in cmd.command

    def test_resume_basic_command(self, runner):
        """Should build basic resume command."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_resume_command(
                session_ref="abc123",
                prompt="Continue"
            )

            assert "resume" in cmd.command
            assert "abc123" in cmd.command
            assert "-" in cmd.command
            assert cmd.prompt == "Continue"
            assert cmd.stream is False  # Resume uses non-streaming

    def test_resume_last_session(self, runner):
        """Should use --last flag for last session."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_resume_command(
                session_ref="--last",
                prompt="Continue"
            )

            assert "--last" in cmd.command

    def test_resume_with_search(self, runner):
        """Resume should also support search flag."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_resume_command(
                session_ref="abc123",
                prompt="Continue",
                enable_search=True
            )

            assert "--enable" in cmd.command

    def test_not_found_hint(self, runner):
        """Should include helpful installation hint."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "Codex CLI" in cmd.not_found_hint

    def test_resume_rejects_flag_injection(self, runner):
        """Should reject session_ref starting with dash to prevent flag injection."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            with pytest.raises(ValueError) as exc_info:
                runner.build_resume_command(
                    session_ref="--malicious-flag",
                    prompt="Hello"
                )

            assert "cannot start with '-'" in str(exc_info.value)

    def test_resume_rejects_single_dash(self, runner):
        """Should reject session_ref that is just a dash."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            with pytest.raises(ValueError) as exc_info:
                runner.build_resume_command(
                    session_ref="-",
                    prompt="Hello"
                )

            assert "cannot start with '-'" in str(exc_info.value)

    def test_resume_accepts_valid_uuid(self, runner):
        """Should accept valid UUID-like session IDs."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_resume_command(
                session_ref="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                prompt="Hello"
            )

            assert "a1b2c3d4-e5f6-7890-abcd-ef1234567890" in cmd.command


class TestGeminiRunner:
    """Tests for Gemini CLI command construction."""

    @pytest.fixture
    def runner(self):
        return GeminiRunner()

    def test_exec_basic_command(self, runner):
        """Should build basic exec command."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert cmd.command[0] == "gemini"
            assert "-p" in cmd.command  # Prompt via -p flag for headless mode
            assert "Hello" in cmd.command
            assert cmd.prompt == ""  # Prompt is in command, not stdin
            assert cmd.output_prefix == "Gemini Output"
            assert cmd.stream is True

    def test_exec_with_working_directory(self, runner):
        """Should add --include-directories flag for working directory."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_exec_command(
                prompt="Hello",
                working_directory="/path/to/dir"
            )

            assert "--include-directories" in cmd.command
            idx = cmd.command.index("--include-directories")
            assert cmd.command[idx + 1] == "/path/to/dir"
            assert cmd.cwd == "/path/to/dir"

    def test_exec_yolo_mode(self, runner):
        """Should add yolo approval mode when enabled."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = True

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--approval-mode" in cmd.command
            idx = cmd.command.index("--approval-mode")
            assert cmd.command[idx + 1] == "yolo"

    def test_exec_without_yolo(self, runner):
        """Should not include yolo flag when disabled."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--approval-mode" not in cmd.command

    def test_resume_basic_command(self, runner):
        """Should build basic resume command."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_resume_command(
                session_ref="latest",
                prompt="Continue"
            )

            assert "-r" in cmd.command
            assert "latest" in cmd.command
            assert "-p" in cmd.command
            assert "Continue" in cmd.command
            assert cmd.prompt == ""  # Prompt is in command via -p flag
            assert cmd.stream is False  # Resume uses non-streaming

    def test_resume_with_yolo(self, runner):
        """Resume should also use yolo mode when enabled."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = True

            cmd = runner.build_resume_command(
                session_ref="latest",
                prompt="Continue"
            )

            assert "--approval-mode" in cmd.command

    def test_not_found_hint(self, runner):
        """Should include helpful installation hint."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "npm install" in cmd.not_found_hint
            assert "@google/gemini-cli" in cmd.not_found_hint

    def test_search_ignored(self, runner):
        """Gemini should accept but ignore enable_search parameter."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            # Should not raise an error
            cmd = runner.build_exec_command(
                prompt="Hello",
                enable_search=True
            )

            # Search is not applicable to Gemini
            assert "--search" not in cmd.command
            assert "--enable" not in cmd.command

    def test_exec_handles_dash_prompt(self, runner):
        """Dash prompts are safe as -p flag argument, not parsed as CLI flags."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_exec_command(prompt="-malicious prompt")

            # Prompt is passed as -p flag's argument, safe from flag parsing
            idx = cmd.command.index("-p")
            assert cmd.command[idx + 1] == "-malicious prompt"

    def test_resume_handles_dash_prompt(self, runner):
        """Dash prompts are safe as -p flag argument in resume mode."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_resume_command(
                session_ref="latest",
                prompt="--dangerous"
            )

            # Prompt is passed as -p flag's argument, safe from flag parsing
            idx = cmd.command.index("-p")
            assert cmd.command[idx + 1] == "--dangerous"


class TestOpenCodeRunner:
    """Tests for OpenCode CLI command construction."""

    @pytest.fixture
    def runner(self):
        return OpenCodeRunner()

    def test_exec_basic_command(self, runner):
        """Should build basic exec command."""
        with patch("owlex.agents.opencode.config") as mock_config:
            mock_config.opencode.model = None
            mock_config.opencode.agent = None
            mock_config.opencode.json_output = False

            cmd = runner.build_exec_command(prompt="Fix the bug")

            assert cmd.command[0] == "opencode"
            assert cmd.command[1] == "run"
            assert "Fix the bug" in cmd.command
            assert cmd.prompt == ""  # Prompt is in command as positional arg
            assert cmd.output_prefix == "OpenCode Output"
            assert cmd.stream is True

    def test_exec_with_working_directory(self, runner):
        """Should set cwd for working directory."""
        with patch("owlex.agents.opencode.config") as mock_config:
            mock_config.opencode.model = None
            mock_config.opencode.agent = None
            mock_config.opencode.json_output = False

            cmd = runner.build_exec_command(
                prompt="Hello",
                working_directory="/path/to/dir"
            )

            assert cmd.cwd == "/path/to/dir"

    def test_exec_with_model(self, runner):
        """Should add model flag when configured."""
        with patch("owlex.agents.opencode.config") as mock_config:
            mock_config.opencode.model = "anthropic/claude-sonnet-4"
            mock_config.opencode.agent = None
            mock_config.opencode.json_output = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--model" in cmd.command
            idx = cmd.command.index("--model")
            assert cmd.command[idx + 1] == "anthropic/claude-sonnet-4"

    def test_exec_with_agent(self, runner):
        """Should add agent flag when configured."""
        with patch("owlex.agents.opencode.config") as mock_config:
            mock_config.opencode.model = None
            mock_config.opencode.agent = "build"
            mock_config.opencode.json_output = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--agent" in cmd.command
            idx = cmd.command.index("--agent")
            assert cmd.command[idx + 1] == "build"

    def test_exec_with_json_output(self, runner):
        """Should add format json flag when configured."""
        with patch("owlex.agents.opencode.config") as mock_config:
            mock_config.opencode.model = None
            mock_config.opencode.agent = None
            mock_config.opencode.json_output = True

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--format" in cmd.command
            idx = cmd.command.index("--format")
            assert cmd.command[idx + 1] == "json"

    def test_resume_with_continue(self, runner):
        """Should use --continue for latest session."""
        with patch("owlex.agents.opencode.config") as mock_config:
            mock_config.opencode.model = None
            mock_config.opencode.agent = None
            mock_config.opencode.json_output = False

            cmd = runner.build_resume_command(
                session_ref="--continue",
                prompt="Continue working"
            )

            assert "--continue" in cmd.command
            assert "Continue working" in cmd.command
            assert cmd.prompt == ""  # Prompt is in command as positional arg
            assert cmd.stream is False  # Resume uses non-streaming

    def test_resume_with_session_id(self, runner):
        """Should use --session for specific session ID."""
        with patch("owlex.agents.opencode.config") as mock_config:
            mock_config.opencode.model = None
            mock_config.opencode.agent = None
            mock_config.opencode.json_output = False

            cmd = runner.build_resume_command(
                session_ref="abc123",
                prompt="Continue"
            )

            assert "--session" in cmd.command
            assert "abc123" in cmd.command

    def test_resume_rejects_flag_injection(self, runner):
        """Should reject session refs that look like flags."""
        with patch("owlex.agents.opencode.config") as mock_config:
            mock_config.opencode.model = None
            mock_config.opencode.agent = None
            mock_config.opencode.json_output = False

            with pytest.raises(ValueError) as exc_info:
                runner.build_resume_command(
                    session_ref="--malicious",
                    prompt="Hello"
                )

            assert "cannot start with '-'" in str(exc_info.value)

    def test_not_found_hint(self, runner):
        """Should include helpful installation hint."""
        with patch("owlex.agents.opencode.config") as mock_config:
            mock_config.opencode.model = None
            mock_config.opencode.agent = None
            mock_config.opencode.json_output = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "opencode.ai" in cmd.not_found_hint

    def test_exec_handles_dash_prompt(self, runner):
        """Should use -- separator to prevent prompts being parsed as flags."""
        with patch("owlex.agents.opencode.config") as mock_config:
            mock_config.opencode.model = None
            mock_config.opencode.agent = None
            mock_config.opencode.json_output = False

            cmd = runner.build_exec_command(prompt="-malicious prompt")

            # -- separator should be present to prevent flag injection
            assert "--" in cmd.command
            assert "-malicious prompt" in cmd.command
            # Verify -- comes before the prompt
            separator_idx = cmd.command.index("--")
            prompt_idx = cmd.command.index("-malicious prompt")
            assert separator_idx < prompt_idx

    def test_resume_handles_dash_prompt(self, runner):
        """Resume should use -- separator to prevent prompts being parsed as flags."""
        with patch("owlex.agents.opencode.config") as mock_config:
            mock_config.opencode.model = None
            mock_config.opencode.agent = None
            mock_config.opencode.json_output = False

            cmd = runner.build_resume_command(
                session_ref="--continue",
                prompt="--dangerous"
            )

            # -- separator should be present to prevent flag injection
            assert cmd.command.count("--") >= 1  # At least one -- for separator
            assert "--dangerous" in cmd.command


class TestAiChatRunner:
    """Tests for AiChat CLI command construction."""

    @pytest.fixture
    def runner(self):
        return AiChatRunner()

    def test_exec_basic_command(self, runner):
        """Should build basic exec command."""
        with patch("owlex.agents.aichat.config") as mock_config:
            mock_config.aichat.model = None

            cmd = runner.build_exec_command(prompt="Hello")

            assert cmd.command[0] == "aichat"
            assert "-s" in cmd.command  # Session flag
            assert cmd.prompt == "Hello"  # Prompt via stdin
            assert cmd.output_prefix == "AiChat Output"
            assert cmd.stream is True

    def test_exec_with_model(self, runner):
        """Should add -m flag when model is configured."""
        with patch("owlex.agents.aichat.config") as mock_config:
            mock_config.aichat.model = "openai:gpt-4o"

            cmd = runner.build_exec_command(prompt="Hello")

            assert "-m" in cmd.command
            idx = cmd.command.index("-m")
            assert cmd.command[idx + 1] == "openai:gpt-4o"

    def test_exec_without_model(self, runner):
        """Should not include -m flag when no model configured."""
        with patch("owlex.agents.aichat.config") as mock_config:
            mock_config.aichat.model = None

            cmd = runner.build_exec_command(prompt="Hello")

            assert "-m" not in cmd.command

    def test_exec_with_working_directory(self, runner):
        """Should set cwd for working directory."""
        with patch("owlex.agents.aichat.config") as mock_config:
            mock_config.aichat.model = None

            cmd = runner.build_exec_command(
                prompt="Hello",
                working_directory="/path/to/dir"
            )

            assert cmd.cwd == "/path/to/dir"

    def test_exec_generates_session_name(self, runner):
        """Should generate a unique session name with owlex- prefix."""
        with patch("owlex.agents.aichat.config") as mock_config:
            mock_config.aichat.model = None

            cmd = runner.build_exec_command(prompt="Hello")

            idx = cmd.command.index("-s")
            session_name = cmd.command[idx + 1]
            assert session_name.startswith("owlex-")

    def test_resume_with_session(self, runner):
        """Should resume with the provided session name."""
        with patch("owlex.agents.aichat.config") as mock_config:
            mock_config.aichat.model = None

            cmd = runner.build_resume_command(
                session_ref="owlex-abc123",
                prompt="Continue"
            )

            assert "-s" in cmd.command
            idx = cmd.command.index("-s")
            assert cmd.command[idx + 1] == "owlex-abc123"
            assert cmd.prompt == "Continue"  # Prompt via stdin
            assert cmd.stream is False  # Resume uses non-streaming

    def test_resume_rejects_flag_injection(self, runner):
        """Should reject session_ref starting with dash to prevent flag injection."""
        with patch("owlex.agents.aichat.config") as mock_config:
            mock_config.aichat.model = None

            with pytest.raises(ValueError) as exc_info:
                runner.build_resume_command(
                    session_ref="--malicious-flag",
                    prompt="Hello"
                )

            assert "cannot start with '-'" in str(exc_info.value)

    def test_not_found_hint(self, runner):
        """Should include helpful installation hint."""
        with patch("owlex.agents.aichat.config") as mock_config:
            mock_config.aichat.model = None

            cmd = runner.build_exec_command(prompt="Hello")

            assert "aichat" in cmd.not_found_hint
            assert "github.com" in cmd.not_found_hint

    def test_exec_handles_dash_prompt(self, runner):
        """Should use stdin to prevent prompts being parsed as flags."""
        with patch("owlex.agents.aichat.config") as mock_config:
            mock_config.aichat.model = None

            cmd = runner.build_exec_command(prompt="-malicious prompt")

            # Prompt should be passed via stdin, not in command
            assert "-malicious prompt" not in cmd.command
            assert cmd.prompt == "-malicious prompt"


class TestAgentInterface:
    """Tests for AgentRunner interface compliance."""

    def test_codex_has_name(self):
        """Codex runner should have name property."""
        runner = CodexRunner()
        assert runner.name == "codex"

    def test_gemini_has_name(self):
        """Gemini runner should have name property."""
        runner = GeminiRunner()
        assert runner.name == "gemini"

    def test_opencode_has_name(self):
        """OpenCode runner should have name property."""
        runner = OpenCodeRunner()
        assert runner.name == "opencode"

    def test_aichat_has_name(self):
        """AiChat runner should have name property."""
        runner = AiChatRunner()
        assert runner.name == "aichat"

    def test_codex_has_output_cleaner(self):
        """Codex runner should provide output cleaner."""
        runner = CodexRunner()
        cleaner = runner.get_output_cleaner()
        assert callable(cleaner)

    def test_gemini_has_output_cleaner(self):
        """Gemini runner should provide output cleaner."""
        runner = GeminiRunner()
        cleaner = runner.get_output_cleaner()
        assert callable(cleaner)

    def test_opencode_has_output_cleaner(self):
        """OpenCode runner should provide output cleaner."""
        runner = OpenCodeRunner()
        cleaner = runner.get_output_cleaner()
        assert callable(cleaner)

    def test_aichat_has_output_cleaner(self):
        """AiChat runner should provide output cleaner."""
        runner = AiChatRunner()
        cleaner = runner.get_output_cleaner()
        assert callable(cleaner)


class TestClaudeORRunner:
    """Tests for ClaudeOR command construction and JSON output handling."""

    @pytest.fixture
    def runner(self):
        return ClaudeORRunner()

    def test_exec_requests_json_output(self, runner):
        """Exec should ask for JSON so the session ID can be read."""
        cmd = runner.build_exec_command(prompt="Hello")

        idx = cmd.command.index("--output-format")
        assert cmd.command[idx + 1] == "json"
        assert cmd.command[-1] == "Hello"

    def test_resume_by_id_requests_json_output(self, runner):
        """Resume should pass the session ID and keep JSON output."""
        cmd = runner.build_resume_command(session_ref="2cf8b2f3-0eb0-49ff-ade5-19d714991538", prompt="Again")

        idx = cmd.command.index("--resume")
        assert cmd.command[idx + 1] == "2cf8b2f3-0eb0-49ff-ade5-19d714991538"
        assert cmd.command[cmd.command.index("--output-format") + 1] == "json"

    def test_cleaner_extracts_result(self):
        """The answer comes from the JSON result field."""
        raw = '{"type":"result","result":"line one\\n\\n\\n\\nline two","session_id":"abc"}'
        with patch("owlex.agents.claudeor.config") as mock_config:
            mock_config.claudeor.clean_output = True
            assert clean_claudeor_output(raw) == "line one\n\nline two"

    def test_cleaner_extracts_result_without_cleaning(self):
        """CLAUDEOR_CLEAN_OUTPUT=false must still not return raw JSON."""
        raw = '{"type":"result","result":"answer\\n","session_id":"abc"}'
        with patch("owlex.agents.claudeor.config") as mock_config:
            mock_config.claudeor.clean_output = False
            assert clean_claudeor_output(raw) == "answer\n"

    def test_cleaner_passes_plain_text_through(self):
        """Non-JSON output is cleaned as before."""
        with patch("owlex.agents.claudeor.config") as mock_config:
            mock_config.claudeor.clean_output = True
            assert clean_claudeor_output("plain\n\n\n\nanswer\n") == "plain\n\nanswer"
