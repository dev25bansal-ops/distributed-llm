"""Exercise the shell's non-interactive surface.

The REPL itself needs a terminal, but every command handler, the @file
expansion and session persistence are plain functions and are tested directly.
"""

from __future__ import annotations

import json

import pytest

from distllm.cli import shell


@pytest.fixture()
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTLLM_SHELL_DIR", str(tmp_path / "sessions"))
    return shell.Session(name="t", model="m", temperature=0.7, max_tokens=64)


class TestSlashCommands:
    def test_help_does_not_exit(self, session):
        assert shell.handle_slash("/help", session) is False

    def test_exit_signals_exit(self, session):
        assert shell.handle_slash("/exit", session) is True

    def test_model_switches_and_reports(self, session, capsys):
        shell.handle_slash("/model new-model", session)
        assert session.model == "new-model"
        shell.handle_slash("/model", session)
        assert "new-model" in capsys.readouterr().out

    def test_temp_is_clamped_to_valid_range(self, session):
        shell.handle_slash("/temp 5.0", session)
        assert session.temperature == 2.0
        shell.handle_slash("/temp -1", session)
        assert session.temperature == 0.0

    def test_temp_rejects_non_numeric(self, session):
        before = session.temperature
        shell.handle_slash("/temp hot", session)
        assert session.temperature == before

    def test_max_tokens_parsed(self, session):
        shell.handle_slash("/max 512", session)
        assert session.max_tokens == 512

    def test_system_prompt_set(self, session):
        shell.handle_slash("/system be terse", session)
        assert session.system_prompt == "be terse"

    def test_clear_empties_history_and_attachments(self, session):
        session.messages.append({"role": "user", "content": "hi"})
        session.attached.append({"path": "x", "content": "y"})
        shell.handle_slash("/clear", session)
        assert session.messages == []
        assert session.attached == []

    def test_unknown_command_is_reported(self, session, capsys):
        shell.handle_slash("/bogus", session)
        assert "Unknown command" in capsys.readouterr().out


class TestFileContext:
    def test_at_file_inlines_contents(self, tmp_path, session):
        f = tmp_path / "notes.txt"
        f.write_text("alpha beta", encoding="utf-8")
        prompt = shell.expand_at_files(f"summarise @{f}", session.attached, shell.console)
        assert "alpha beta" in prompt
        assert f"[attached: {f}]" in prompt
        assert len(session.attached) == 1

    def test_missing_file_is_reported_not_fatal(self, session, capsys):
        prompt = shell.expand_at_files("see @/nope/missing.txt", session.attached, shell.console)
        assert "No such file" in capsys.readouterr().out
        assert session.attached == []
        assert prompt == "see @/nope/missing.txt"

    def test_slash_file_attaches_for_next_turn(self, tmp_path, session):
        f = tmp_path / "a.py"
        f.write_text("print(1)", encoding="utf-8")
        shell.handle_slash(f"/file {f}", session)
        assert len(session.attached) == 1

    def test_large_file_is_truncated(self, tmp_path, session, capsys):
        big = tmp_path / "big.txt"
        big.write_text("x" * (shell.MAX_FILE_BYTES + 5000), encoding="utf-8")
        shell.handle_slash(f"/file {big}", session)
        assert "truncated" in capsys.readouterr().out
        assert len(session.attached[0]["content"]) < shell.MAX_FILE_BYTES + 1000

    def test_directory_lists_entries(self, tmp_path, session):
        (tmp_path / "one.py").write_text("", encoding="utf-8")
        loaded = shell.read_file_for_context(str(tmp_path))
        assert loaded is not None
        assert "one.py" in loaded["content"]

    def test_email_address_is_not_treated_as_a_file(self, session, capsys):
        prompt = shell.expand_at_files("mail me at a@b.com", session.attached, shell.console)
        assert prompt == "mail me at a@b.com"
        assert session.attached == []


class TestSessions:
    def test_save_then_load_round_trips(self, tmp_path, monkeypatch, session):
        monkeypatch.setenv("DISTLLM_SHELL_DIR", str(tmp_path / "s"))
        session.messages.append({"role": "user", "content": "hello"})
        session.prompt_tokens = 11
        path = shell.save_session(session, "keep")
        assert path.exists()

        restored = shell.load_session("keep")
        assert restored.messages == session.messages
        assert restored.prompt_tokens == 11
        assert restored.model == session.model

    def test_save_uses_a_safe_filename(self, tmp_path, monkeypatch, session):
        monkeypatch.setenv("DISTLLM_SHELL_DIR", str(tmp_path / "s"))
        path = shell.save_session(session, "../../etc/passwd")
        assert path.parent == tmp_path / "s"
        assert ".." not in path.name

    def test_load_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DISTLLM_SHELL_DIR", str(tmp_path / "s"))
        with pytest.raises(FileNotFoundError):
            shell.load_session("absent")

    def test_list_sessions_reports_contents(self, tmp_path, monkeypatch, session):
        monkeypatch.setenv("DISTLLM_SHELL_DIR", str(tmp_path / "s"))
        session.messages.append({"role": "user", "content": "x"})
        shell.save_session(session, "listed")
        names = [n for n, _, _ in shell.list_sessions()]
        assert "listed" in names

    def test_saved_file_is_valid_json(self, tmp_path, monkeypatch, session):
        monkeypatch.setenv("DISTLLM_SHELL_DIR", str(tmp_path / "s"))
        path = shell.save_session(session, "jsoncheck")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["model"] == session.model


class TestMessageBuilding:
    def test_system_prompt_prepended(self, session):
        session.system_prompt = "be brief"
        session.messages.append({"role": "user", "content": "hi"})
        messages = shell.build_messages(session)
        assert messages[0] == {"role": "system", "content": "be brief"}
        assert messages[1]["content"] == "hi"

    def test_no_system_prompt_means_no_system_message(self, session):
        session.messages.append({"role": "user", "content": "hi"})
        assert shell.build_messages(session) == [{"role": "user", "content": "hi"}]

    def test_completion_lists_command_names(self):
        names = shell.slash_command_names()
        assert "/help" in names and "/exit" in names
        assert all(n.startswith("/") and " " not in n for n in names)


class TestTerminalFallback:
    """prompt_toolkit refuses to start under git-bash on Windows.

    It raises "Found xterm-256color, while expecting a Windows console", which
    made `distllm shell` unusable in exactly the terminal a Windows developer
    is most likely to use. The shell must degrade, not fail.
    """

    def test_msys_uses_the_plain_prompt(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("MSYSTEM", "MINGW64")
        monkeypatch.setenv("TERM", "xterm-256color")
        session_obj = shell._make_prompt_session(tmp_path / "h.txt", shell.console)
        assert isinstance(session_obj, shell._PlainPrompt)
        assert "Simplified prompt" in capsys.readouterr().out

    def test_plain_prompt_returns_typed_line(self, tmp_path, monkeypatch):
        prompt = shell._PlainPrompt(tmp_path / "h.txt")
        monkeypatch.setattr("builtins.input", lambda _msg="": "/help")
        assert prompt.prompt("> ") == "/help"

    def test_plain_prompt_propagates_eof(self, tmp_path, monkeypatch):
        prompt = shell._PlainPrompt(tmp_path / "h.txt")

        def _eof(_msg=""):
            raise EOFError

        monkeypatch.setattr("builtins.input", _eof)
        with pytest.raises(EOFError):
            prompt.prompt("> ")