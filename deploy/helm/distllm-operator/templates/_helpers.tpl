{{- define "distllm-operator.name" -}}
{{- default .Chart.Name .Values.operator.name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "distllm-operator.fullname" -}}
{{- $name := default .Chart.Name .Values.operator.name }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "distllm-operator.labels" -}}
app.kubernetes.io/name: {{ include "distllm-operator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: operator
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{- define "distllm-operator.coordinator-selector-labels" -}}
app.kubernetes.io/name: {{ include "distllm-operator.name" . }}
app.kubernetes.io/component: coordinator
{{- end }}

{{- define "distllm-operator.worker-selector-labels" -}}
app.kubernetes.io/name: {{ include "distllm-operator.name" . }}
app.kubernetes.io/component: worker
{{- end }}
