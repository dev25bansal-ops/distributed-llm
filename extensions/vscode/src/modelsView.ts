import * as vscode from "vscode";
import { fetchModels, ModelInfo } from "./modelsApi";

// ---------------------------------------------------------------------------
// Models view (TreeView)
// ---------------------------------------------------------------------------

type ModelItemKind = "model" | "error" | "empty" | "loading";

export class ModelItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState,
    public readonly kind: ModelItemKind,
    public readonly modelId?: string,
  ) {
    super(label, collapsibleState);

    if (kind === "model" && modelId) {
      this.description = "click to set as default";
      this.contextValue = "distllmModel";
      this.iconPath = new vscode.ThemeIcon("symbol-namespace");
      this.command = {
        command: "distllm.setModel",
        title: "Use Model",
        arguments: [modelId],
      };
    } else if (kind === "error") {
      this.iconPath = new vscode.ThemeIcon("error");
    } else if (kind === "empty") {
      this.iconPath = new vscode.ThemeIcon("info");
    } else if (kind === "loading") {
      this.iconPath = new vscode.ThemeIcon("loading~spin");
    }
  }
}

export class ModelsViewProvider implements vscode.TreeDataProvider<ModelItem> {
  private readonly _onDidChangeTreeData = new vscode.EventEmitter<void>();
  public readonly onDidChangeTreeData: vscode.Event<void> = this._onDidChangeTreeData.event;

  private _items: ModelItem[] = [
    new ModelItem("Loading…", vscode.TreeItemCollapsibleState.None, "loading"),
  ];
  private _loading = false;

  /** Re-fetch the model list and refresh the view. */
  public refresh(): void {
    void this.load();
  }

  /** Initial / manual load of models from the API. */
  public async load(): Promise<void> {
    if (this._loading) {
      return;
    }
    // Refuse to contact a workspace-controlled apiUrl from an untrusted
    // workspace — this is the models-tree egress path and must honor the
    // same trust boundary as sendSelection/openDashboard/fetchHealth/etc.
    if (!vscode.workspace.isTrusted) {
      this._items = [
        new ModelItem(
          "DistLLM: workspace untrusted; refusing to list models",
          vscode.TreeItemCollapsibleState.None,
          "error",
        ),
      ];
      this._onDidChangeTreeData.fire();
      return;
    }
    this._loading = true;
    const cfg = vscode.workspace.getConfiguration("distllm");
    const apiUrl = cfg.get<string>("apiUrl", "http://localhost:8000");

    try {
      const models: ModelInfo[] = await fetchModels(apiUrl);
      if (models.length === 0) {
        this._items = [
          new ModelItem("No models available", vscode.TreeItemCollapsibleState.None, "empty"),
        ];
      } else {
        this._items = models.map(
          (m) => new ModelItem(m.id, vscode.TreeItemCollapsibleState.None, "model", m.id),
        );
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this._items = [
        new ModelItem(`Error: ${msg}`, vscode.TreeItemCollapsibleState.None, "error"),
      ];
    } finally {
      this._loading = false;
      this._onDidChangeTreeData.fire();
    }
  }

  public getTreeItem(element: ModelItem): vscode.TreeItem {
    return element;
  }

  public getChildren(element?: ModelItem): ModelItem[] {
    if (element) {
      return [];
    }
    return this._items;
  }
}
