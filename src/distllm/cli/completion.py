"""Shell completion generation for distllm CLI.

Extracted from :mod:`distllm.cli.main`.
"""

from __future__ import annotations

import subprocess
import sys

import typer
from rich.console import Console

console = Console()


def completion_command(shell: str) -> None:
    """Generate shell autocomplete scripts for distllm CLI.

    Usage:
        distllm completion bash >> ~/.bashrc
        distllm completion zsh >> ~/.zshrc
        distllm completion fish > ~/.config/fish/completions/distllm.fish
        distllm completion powershell >> $PROFILE

    Then restart your shell or source the file.
    """
    shell_map = {
        "bash": "bash",
        "zsh": "zsh",
        "fish": "fish",
        "powershell": "powershell",
    }

    if shell not in shell_map:
        console.print(f"[red]Unsupported shell: {shell}[/red]")
        console.print(f"Supported: {', '.join(shell_map.keys())}")
        raise typer.Exit(1)

    # Typer provides built-in completion via the --show-completion flag
    if shell == "bash":
        script = """
# DistLLM CLI autocomplete for bash
_evalcache() { eval "$1"; }
_distllm_bash_complete() {
    local IFS=$'\\n'
    COMPREPLY=( $(compgen -W "$(distllm --show-completion bash)" -- "${COMP_WORDS[COMP_CWORD]}") )
}
complete -F _distllm_bash_complete distllm
"""
    elif shell == "zsh":
        script = """# DistLLM CLI autocomplete for zsh
#compdef distllm
_distllm() {
    compadd $(distllm --show-completion zsh)
}
_distllm
"""
    elif shell == "fish":
        script = """# DistLLM CLI autocomplete for fish
complete -c distllm -f
complete -c distllm -a '(distllm --show-completion fish)'
"""
    elif shell == "powershell":
        script = """# DistLLM CLI autocomplete for PowerShell
Register-ArgumentCompleter -Native -CommandName distllm -ScriptBlock {
    param($commandName, $wordToComplete, $cursorPosition)
    distllm --show-completion powershell | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
}
"""
    else:
        script = ""

    console.print(f"[green]Generated {shell} completion script:[/green]")
    console.print()
    console.print(script)
    console.print(f"[dim]Add this to your shell config file to enable autocomplete.[/dim]")
