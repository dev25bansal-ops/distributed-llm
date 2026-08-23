export interface ConfigIssue {
    setting: string;
    message: string;
}
export declare function validateDistllmConfig(): ConfigIssue[];
export declare function showConfigWarnings(): void;
