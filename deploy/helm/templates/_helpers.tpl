{{/*
Expand the name of the chart.
*/}}
{{- define "distllm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "distllm.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "distllm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "distllm.labels" -}}
helm.sh/chart: {{ include "distllm.chart" . }}
{{ include "distllm.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "distllm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "distllm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service account name
*/}}
{{- define "distllm.serviceAccountName" -}}
{{ include "distllm.fullname" . }}
{{- end }}

{{/*
GPU resource name
*/}}
{{- define "distllm.gpuResourceName" -}}
{{- default "nvidia.com/gpu" . }}
{{- end }}

{{/*
Coordinator labels
*/}}
{{- define "distllm.coordinatorLabels" -}}
{{ include "distllm.selectorLabels" . }}
app.distllm.io/component: coordinator
{{- end }}

{{/*
Worker labels for a given pool
Usage: include "distllm.workerLabels" (dict "root" $ "pool" $pool)
*/}}
{{- define "distllm.workerLabels" -}}
{{ include "distllm.selectorLabels" .root }}
app.distllm.io/component: worker
app.distllm.io/pool: {{ .pool.name }}
{{- end }}
