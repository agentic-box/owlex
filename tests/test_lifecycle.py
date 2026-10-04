"""
Integration tests for task lifecycle management.
Tests run_command, timeout handling, and task cancellation.
"""

import asyncio
import os
import sys
from unittest.mock import patch

import pytest
from owlex.engine import TaskEngine, resolve_executable
from owlex.models import TaskStatus


class TestTaskExecution:
    """Tests for basic task execution."""

    async def test_successful_execution(self, engine, helper_script_dir):
        """Task should complete successfully with correct output."""
        task = engine.create_task(
            command="test_echo",
            args={"prompt": "hello"},
        )

        await engine.run_command(
            task=task,
            command=[sys.executable, str(helper_script_dir / "echo_stdin.py")],
            prompt="hello world",
            output_cleaner=lambda x, p: x.strip(),
            output_prefix="Test Output",
            timeout=5,
            stream=False,
        )

        assert task.status == TaskStatus.COMPLETED.value
        assert "hello world" in task.result
        assert task.error is None
        assert task.completion_time is not None

    async def test_execution_with_error(self, engine, helper_script_dir):
        """Task should fail when subprocess exits with error."""
        task = engine.create_task(
            command="test_error",
            args={},
        )

        await engine.run_command(
            task=task,
            command=[sys.executable, str(helper_script_dir / "exit_error.py")],
            prompt="",
            output_cleaner=lambda x, p: x.strip(),
            output_prefix="Test Output",
            timeout=5,
            stream=False,
        )

        assert task.status == TaskStatus.FAILED.value
        assert task.error is not None
        assert "exit code 1" in task.error
        assert task.completion_time is not None

    async def test_command_not_found(self, engine):
        """Task should fail gracefully when command doesn't exist."""
        task = engine.create_task(
            command="test_not_found",
            args={},
        )

        await engine.run_command(
            task=task,
            command=["nonexistent_command_12345"],
            prompt="",
            output_cleaner=lambda x, p: x,
            output_prefix="Test",
            timeout=5,
            not_found_hint="Install the command first.",
        )

        assert task.status == TaskStatus.FAILED.value
        assert "not found" in task.error.lower()
        assert "Install the command first" in task.error


class TestResolveExecutable:
    """Tests for argv[0] resolution (Windows .cmd shims)."""

    def test_resolves_name_via_which(self):
        """Should replace argv[0] with the path shutil.which finds."""
        with patch("owlex.engine.shutil.which", return_value="/npm/codex.CMD"):
            assert resolve_executable(["codex", "exec", "-"]) == ["/npm/codex.CMD", "exec", "-"]

    def test_unresolved_name_passes_through(self):
        """Should leave an unknown name unchanged so the not-found hint still fires."""
        with patch("owlex.engine.shutil.which", return_value=None):
            assert resolve_executable(["missing", "--version"]) == ["missing", "--version"]

    def test_empty_command(self):
        """Should return an empty command unchanged."""
        assert resolve_executable([]) == []

    @pytest.mark.skipif(sys.platform != "win32", reason="PATHEXT shims are Windows-only")
    async def test_runs_cmd_shim_on_path(self, engine, tmp_path, monkeypatch):
        """A CLI installed only as a .cmd shim (as npm does) should run by bare name."""
        (tmp_path / "owlexshim.cmd").write_text("@echo shim ran\n")
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

        task = engine.create_task(command="test_shim", args={})
        await engine.run_command(
            task=task,
            command=["owlexshim"],
            prompt="",
            output_cleaner=lambda x, p: x.strip(),
            output_prefix="Test",
            timeout=10,
        )

        assert task.status == TaskStatus.COMPLETED.value, task.error
        assert "shim ran" in task.result


class TestTimeoutHandling:
    """Tests for timeout behavior."""

    async def test_timeout_kills_process(self, engine, helper_script_dir):
        """Timed out task should have its subprocess killed."""
        task = engine.create_task(
            command="test_timeout",
            args={},
        )

        await engine.run_command(
            task=task,
            command=[sys.executable, str(helper_script_dir / "sleep_forever.py")],
            prompt="",
            output_cleaner=lambda x, p: x,
            output_prefix="Test",
            timeout=1,  # 1 second timeout
            stream=False,
        )

        assert task.status == TaskStatus.FAILED.value
        assert "timed out" in task.error.lower()
        assert task.completion_time is not None
        # Process should be killed (returncode set)
        assert task.process.returncode is not None

    async def test_timeout_streaming_mode(self, engine, helper_script_dir):
        """Timeout should work correctly in streaming mode."""
        task = engine.create_task(
            command="test_timeout_stream",
            args={},
        )

        await engine.run_command(
            task=task,
            command=[sys.executable, str(helper_script_dir / "sleep_forever.py")],
            prompt="",
            output_cleaner=lambda x, p: x,
            output_prefix="Test",
            timeout=1,
            stream=True,
        )

        assert task.status == TaskStatus.FAILED.value
        assert "timed out" in task.error.lower()
        assert task.process.returncode is not None


class TestTaskCancellation:
    """Tests for task cancellation."""

    async def test_cancel_running_task(self, engine, helper_script_dir):
        """Cancelling a running task should kill subprocess."""
        task = engine.create_task(
            command="test_cancel",
            args={},
        )

        # Start task in background
        run_task = asyncio.create_task(
            engine.run_command(
                task=task,
                command=[sys.executable, str(helper_script_dir / "sleep_forever.py")],
                prompt="",
                output_cleaner=lambda x, p: x,
                output_prefix="Test",
                timeout=60,
                stream=False,
            )
        )

        # Wait for task to start
        await asyncio.sleep(0.2)
        assert task.status == TaskStatus.RUNNING.value

        # Cancel the task
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

        assert task.status == TaskStatus.CANCELLED.value
        assert task.process.returncode is not None


class TestStdinHandling:
    """Tests for stdin handling in streaming vs non-streaming modes."""

    async def test_stdin_non_streaming(self, engine, helper_script_dir):
        """Non-streaming mode should pass stdin via communicate()."""
        task = engine.create_task(
            command="test_stdin_nonstream",
            args={},
        )

        await engine.run_command(
            task=task,
            command=[sys.executable, str(helper_script_dir / "echo_stdin.py")],
            prompt="test input data",
            output_cleaner=lambda x, p: x.strip(),
            output_prefix="Output",
            timeout=5,
            stream=False,
        )

        assert task.status == TaskStatus.COMPLETED.value
        assert "test input data" in task.result

    async def test_stdin_streaming(self, engine, helper_script_dir):
        """Streaming mode should write stdin before reading output."""
        task = engine.create_task(
            command="test_stdin_stream",
            args={},
        )

        await engine.run_command(
            task=task,
            command=[sys.executable, str(helper_script_dir / "echo_stdin.py")],
            prompt="streaming test data",
            output_cleaner=lambda x, p: x.strip(),
            output_prefix="Output",
            timeout=5,
            stream=True,
        )

        assert task.status == TaskStatus.COMPLETED.value
        assert "streaming test data" in task.result

    async def test_empty_stdin(self, engine, helper_script_dir):
        """Should handle empty stdin correctly."""
        task = engine.create_task(
            command="test_empty_stdin",
            args={},
        )

        # Script that outputs without reading stdin
        await engine.run_command(
            task=task,
            command=[sys.executable, str(helper_script_dir / "delayed_output.py")],
            prompt="",
            output_cleaner=lambda x, p: x.strip(),
            output_prefix="Output",
            timeout=5,
            stream=False,
        )

        assert task.status == TaskStatus.COMPLETED.value
        assert "delayed output" in task.result


class TestTaskRegistry:
    """Tests for task registration and retrieval."""

    def test_create_task(self, engine):
        """create_task should register task with unique ID."""
        task = engine.create_task(
            command="test",
            args={"key": "value"},
        )

        assert task.task_id is not None
        assert task.status == TaskStatus.PENDING.value
        assert task.command == "test"
        assert task.args == {"key": "value"}
        assert engine.get_task(task.task_id) is task

    def test_get_nonexistent_task(self, engine):
        """get_task should return None for unknown ID."""
        result = engine.get_task("nonexistent-id")
        assert result is None

    def test_multiple_tasks(self, engine):
        """Should track multiple tasks independently."""
        task1 = engine.create_task(command="cmd1", args={})
        task2 = engine.create_task(command="cmd2", args={})

        assert task1.task_id != task2.task_id
        assert engine.get_task(task1.task_id) is task1
        assert engine.get_task(task2.task_id) is task2
        assert len(engine.tasks) == 2
