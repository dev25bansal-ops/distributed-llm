"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.validateDistllmConfig = validateDistllmConfig;
exports.showConfigWarnings = showConfigWarnings;
const vscode = __importStar(require("vscode"));
function validateDistllmConfig() {
    const cfg = vscode.workspace.getConfiguration("distllm");
    const issues = [];
    // apiUrl must be a valid http(s) URL
    const apiUrl = cfg.get("apiUrl", "");
    try {
        const u = new URL(apiUrl);
        if (u.protocol !== "http:" && u.protocol !== "https:") {
            issues.push({
                setting: "apiUrl",
                message: `distllm.apiUrl must be an http(s) URL, got "${apiUrl}"`,
            });
        }
        else if (u.protocol === "http:") {
            // Plain http is only allowed for loopback hosts (localhost / 127.0.0.1),
            // since an untrusted workspace could otherwise redirect editor data to a
            // remote, unencrypted endpoint.
            const host = u.hostname.toLowerCase();
            const isLocalhost = host === "localhost" ||
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
    }
    catch {
        issues.push({
            setting: "apiUrl",
            message: `distllm.apiUrl is not a valid URL: "${apiUrl}"`,
        });
    }
    // refreshInterval within [2, 300]
    const refreshInterval = cfg.get("refreshInterval", 10);
    if (typeof refreshInterval !== "number" ||
        isNaN(refreshInterval) ||
        refreshInterval < 2 ||
        refreshInterval > 300) {
        issues.push({
            setting: "refreshInterval",
            message: `distllm.refreshInterval must be between 2 and 300 (seconds), got ${refreshInterval}`,
        });
    }
    // maxTokens > 0
    const maxTokens = cfg.get("maxTokens", 256);
    if (typeof maxTokens !== "number" || isNaN(maxTokens) || maxTokens <= 0) {
        issues.push({
            setting: "maxTokens",
            message: `distllm.maxTokens must be greater than 0, got ${maxTokens}`,
        });
    }
    // temperature in [0, 2]
    const temperature = cfg.get("temperature", 0.7);
    if (typeof temperature !== "number" ||
        isNaN(temperature) ||
        temperature < 0 ||
        temperature > 2) {
        issues.push({
            setting: "temperature",
            message: `distllm.temperature must be between 0 and 2, got ${temperature}`,
        });
    }
    return issues;
}
function showConfigWarnings() {
    const issues = validateDistllmConfig();
    if (issues.length === 0) {
        return;
    }
    const bullets = issues.map((i) => `• ${i.message}`).join("\n");
    vscode.window.showWarningMessage(`DistLLM: invalid configuration detected:\n${bullets}`);
}
//# sourceMappingURL=configValidation.js.map