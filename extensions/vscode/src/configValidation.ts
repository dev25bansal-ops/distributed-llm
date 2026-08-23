import * as vscode from "vscode";

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------
// Validates the distllm.* settings contributed in package.json and surfaces
// any problems through a single non-blocking warning message.

export interface ConfigIssue {
  setting: string;
  message: string;
}

export function validateDistllmConfig(): ConfigIssue[] {
  const cfg = vscode.workspace.getConfiguration("distllm");
  const issues: ConfigIssue[] = [];

  // apiUrl must be a valid http(s) URL
  const apiUrl = cfg.get<string>("apiUrl", "");
  try {
    const u = new URL(apiUrl);
    if (u.protocol !== "http:" && u.protocol !== "https:") {
      issues.push({
        setting: "apiUrl",
        message: `distllm.apiUrl must be an http(s) URL, got "${apiUrl}"`,
      });
    } else if (u.protocol === "http:") {
      // Plain http is only allowed for loopback hosts (localhost / 127.0.0.1),
      // since an untrusted workspace could otherwise redirect editor data to a
      // remote, unencrypted endpoint.
      const host = u.hostname.toLowerCase();
      const isLocalhost =
        host === "localhost" ||
        host === "127.0.0.1" ||
        host === "::1" ||
        host.endsWith(".localhost");
      if (!isLocalhost) {
        issues.push({
          setting: "apiUrl",
          message: `distllm.apiUrl must use https for non-localhost hosts, got "${apiUrl}"`,
        });
      }
    }
  } catch {
    issues.push({
      setting: "apiUrl",
      message: `distllm.apiUrl is not a valid URL: "${apiUrl}"`,
    });
  }

  // refreshInterval within [2, 300]
  const refreshInterval = cfg.get<number>("refreshInterval", 10);
  if (
    typeof refreshInterval !== "number" ||
    isNaN(refreshInterval) ||
    refreshInterval < 2 ||
    refreshInterval > 300
  ) {
    issues.push({
      setting: "refreshInterval",
      message: `distllm.refreshInterval must be between 2 and 300 (seconds), got ${refreshInterval}`,
    });
  }

  // maxTokens > 0
  const maxTokens = cfg.get<number>("maxTokens", 256);
  if (typeof maxTokens !== "number" || isNaN(maxTokens) || maxTokens <= 0) {
    issues.push({
      setting: "maxTokens",
      message: `distllm.maxTokens must be greater than 0, got ${maxTokens}`,
    });
  }

  // temperature in [0, 2]
  const temperature = cfg.get<number>("temperature", 0.7);
  if (
    typeof temperature !== "number" ||
    isNaN(temperature) ||
    temperature < 0 ||
    temperature > 2
  ) {
    issues.push({
      setting: "temperature",
      message: `distllm.temperature must be between 0 and 2, got ${temperature}`,
    });
  }

  return issues;
}

export function showConfigWarnings(): void {
  const issues = validateDistllmConfig();
  if (issues.length === 0) {
    return;
  }
  const bullets = issues.map((i) => `• ${i.message}`).join("\n");
  vscode.window.showWarningMessage(`DistLLM: invalid configuration detected:\n${bullets}`);
}
