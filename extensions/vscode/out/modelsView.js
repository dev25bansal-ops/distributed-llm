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
exports.ModelsViewProvider = exports.ModelItem = void 0;
const vscode = __importStar(require("vscode"));
const modelsApi_1 = require("./modelsApi");
class ModelItem extends vscode.TreeItem {
    constructor(label, collapsibleState, kind, modelId) {
        super(label, collapsibleState);
        this.label = label;
        this.collapsibleState = collapsibleState;
        this.kind = kind;
        this.modelId = modelId;
        if (kind === "model" && modelId) {
            this.description = "click to set as default";
            this.contextValue = "distllmModel";
            this.iconPath = new vscode.ThemeIcon("symbol-namespace");
            this.command = {
                command: "distllm.setModel",
                title: "Use Model",
                arguments: [modelId],
            };
        }
        else if (kind === "error") {
            this.iconPath = new vscode.ThemeIcon("error");
        }
        else if (kind === "empty") {
            this.iconPath = new vscode.ThemeIcon("info");
        }
        else if (kind === "loading") {
            this.iconPath = new vscode.ThemeIcon("loading~spin");
        }
    }
}
exports.ModelItem = ModelItem;
class ModelsViewProvider {
    constructor() {
        this._onDidChangeTreeData = new vscode.EventEmitter();
        this.onDidChangeTreeData = this._onDidChangeTreeData.event;
        this._items = [
            new ModelItem("Loading…", vscode.TreeItemCollapsibleState.None, "loading"),
        ];
        this._loading = false;
    }
    /** Re-fetch the model list and refresh the view. */
    refresh() {
        void this.load();
    }
    /** Initial / manual load of models from the API. */
    async load() {
        if (this._loading) {
            return;
        }
        // Refuse to contact a workspace-controlled apiUrl from an untrusted
        // workspace — this is the models-tree egress path and must honor the
        // same trust boundary as sendSelection/openDashboard/fetchHealth/etc.
        if (!vscode.workspace.isTrusted) {
            this._items = [
                new ModelItem("DistLLM: workspace untrusted; refusing to list models", vscode.TreeItemCollapsibleState.None, "error"),
            ];
            this._onDidChangeTreeData.fire();
            return;
        }
        this._loading = true;
        const cfg = vscode.workspace.getConfiguration("distllm");
        const apiUrl = cfg.get("apiUrl", "http://localhost:8000");
        try {
            const models = await (0, modelsApi_1.fetchModels)(apiUrl);
            if (models.length === 0) {
                this._items = [
                    new ModelItem("No models available", vscode.TreeItemCollapsibleState.None, "empty"),
                ];
            }
            else {
                this._items = models.map((m) => new ModelItem(m.id, vscode.TreeItemCollapsibleState.None, "model", m.id));
            }
        }
        catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            this._items = [
                new ModelItem(`Error: ${msg}`, vscode.TreeItemCollapsibleState.None, "error"),
            ];
        }
        finally {
            this._loading = false;
            this._onDidChangeTreeData.fire();
        }
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        if (element) {
            return [];
        }
        return this._items;
    }
}
exports.ModelsViewProvider = ModelsViewProvider;
//# sourceMappingURL=modelsView.js.map