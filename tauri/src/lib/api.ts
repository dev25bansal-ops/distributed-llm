import { invoke } from "@tauri-apps/api/core";
import type {
  ClusterStatus,
  GpuInfo,
  ModelInfo,
  InviteInfo,
  SystemInfo,
} from "./types";

export async function createCluster(
  port?: number,
  model?: string,
): Promise<ClusterStatus> {
  return invoke<ClusterStatus>("create_cluster", { port, model });
}

export async function joinCluster(
  host: string,
  port: number,
): Promise<ClusterStatus> {
  return invoke<ClusterStatus>("join_cluster", { host, port });
}

export async function leaveCluster(): Promise<void> {
  return invoke<void>("leave_cluster");
}

export async function getClusterStatus(): Promise<ClusterStatus> {
  return invoke<ClusterStatus>("get_cluster_status");
}

export async function getGpuMetrics(): Promise<GpuInfo[]> {
  return invoke<GpuInfo[]>("get_gpu_metrics");
}

export async function listModels(): Promise<ModelInfo[]> {
  return invoke<ModelInfo[]>("list_models");
}

export async function downloadModel(modelId: string): Promise<string> {
  return invoke<string>("download_model", { modelId });
}

export async function generateInvite(): Promise<InviteInfo> {
  return invoke<InviteInfo>("generate_invite");
}

export async function getSystemInfo(): Promise<SystemInfo> {
  return invoke<SystemInfo>("get_system_info");
}
