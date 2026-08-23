import * as vscode from "vscode";
type ModelItemKind = "model" | "error" | "empty" | "loading";
export declare class ModelItem extends vscode.TreeItem {
    readonly label: string;
    readonly collapsibleState: vscode.TreeItemCollapsibleState;
    readonly kind: ModelItemKind;
    readonly modelId?: string | undefined;
    constructor(label: string, collapsibleState: vscode.TreeItemCollapsibleState, kind: ModelItemKind, modelId?: string | undefined);
}
export declare class ModelsViewProvider implements vscode.TreeDataProvider<ModelItem> {
    private readonly _onDidChangeTreeData;
    readonly onDidChangeTreeData: vscode.Event<void>;
    private _items;
    private _loading;
    /** Re-fetch the model list and refresh the view. */
    refresh(): void;
    /** Initial / manual load of models from the API. */
    load(): Promise<void>;
    getTreeItem(element: ModelItem): vscode.TreeItem;
    getChildren(element?: ModelItem): ModelItem[];
}
export {};
