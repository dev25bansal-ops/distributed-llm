{{/*
Expand the name of the chart.
*/}}
{{- define "distributed-llm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "distributed-llm.fullname" -}}
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
{{- define "distributed-llm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "distributed-llm.labels" -}}
helm.sh/chart: {{ include "distributed-llm.chart" . }}
{{ include "distributed-llm.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "distributed-llm.selectorLabels" -}}
app.kubernetes.io/name: {{ include "distributed-llm.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
GPU resource name
*/}}
{{- define "distributed-llm.gpuResourceName" -}}
{{- default "nvidia.com/gpu" . }}
{{- end }}

{{/*
Coordinator labels
*/}}
{{- define "distributed-llm.coordinatorLabels" -}}
{{ include "distributed-llm.selectorLabels" . }}
app.distllm.io/component: coordinator
{{- end }}

{{/*
Worker labels for a given pool
Usage: include "distributed-llm.workerLabels" (dict "root" $ "pool" $pool)
*/}}
{{- define "distributed-llm.workerLabels" -}}
{{ include "distributed-llm.selectorLabels" .root }}
app.distllm.io/component: worker
app.distllm.io/pool: {{ .pool.name }}
{{- end }}
