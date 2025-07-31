{{/*
Expand the name of the chart.
*/}}
{{- define "agentic-startup-studio.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "agentic-startup-studio.fullname" -}}
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
{{- define "agentic-startup-studio.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "agentic-startup-studio.labels" -}}
helm.sh/chart: {{ include "agentic-startup-studio.chart" . }}
{{ include "agentic-startup-studio.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: terragon-platform
environment: {{ .Values.global.environment }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "agentic-startup-studio.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agentic-startup-studio.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "agentic-startup-studio.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "agentic-startup-studio.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
AI Agent Worker labels
*/}}
{{- define "agentic-startup-studio.workerLabels" -}}
helm.sh/chart: {{ include "agentic-startup-studio.chart" . }}
{{ include "agentic-startup-studio.workerSelectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: terragon-platform
environment: {{ .Values.global.environment }}
{{- end }}

{{/*
AI Agent Worker selector labels
*/}}
{{- define "agentic-startup-studio.workerSelectorLabels" -}}
app.kubernetes.io/name: {{ .Values.aiAgentWorker.name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create database connection string
*/}}
{{- define "agentic-startup-studio.databaseUrl" -}}
{{- if .Values.postgresql.enabled }}
postgresql://{{ .Values.postgresql.auth.username }}:$(POSTGRES_PASSWORD)@{{ include "agentic-startup-studio.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- else }}
{{- .Values.externalDatabase.url }}
{{- end }}
{{- end }}

{{/*
Create Redis connection string
*/}}
{{- define "agentic-startup-studio.redisUrl" -}}
{{- if .Values.redis.enabled }}
redis://{{ include "agentic-startup-studio.fullname" . }}-redis-master:6379
{{- else }}
{{- .Values.externalRedis.url }}
{{- end }}
{{- end }}

{{/*
Generate certificates secret name
*/}}
{{- define "agentic-startup-studio.certificateSecretName" -}}
{{- if .Values.ingress.tls }}
{{- range .Values.ingress.tls }}
{{- .secretName }}
{{- end }}
{{- else }}
{{- printf "%s-tls" (include "agentic-startup-studio.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "agentic-startup-studio.image" -}}
{{- $registryName := .Values.global.imageRegistry -}}
{{- $repositoryName := .Values.app.image.repository -}}
{{- $tag := .Values.app.image.tag | default .Values.global.imageTag | toString -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else }}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end }}
{{- end }}

{{/*
Return the proper storage class
*/}}
{{- define "agentic-startup-studio.storageClass" -}}
{{- if .Values.global.storageClass }}
{{- printf "storageClassName: %s" .Values.global.storageClass }}
{{- else if .Values.persistence.storageClass }}
{{- printf "storageClassName: %s" .Values.persistence.storageClass }}
{{- end }}
{{- end }}

{{/*
Validate required values
*/}}
{{- define "agentic-startup-studio.validateValues" -}}
{{- if and .Values.postgresql.enabled (not .Values.postgresql.auth.password) }}
{{- fail "PostgreSQL password is required when PostgreSQL is enabled" }}
{{- end }}
{{- if and .Values.monitoring.grafana.enabled (not .Values.monitoring.grafana.adminPassword) }}
{{- fail "Grafana admin password is required when Grafana is enabled" }}
{{- end }}
{{- if not .Values.app.secrets.jwt_secret_key }}
{{- fail "JWT secret key is required" }}
{{- end }}
{{- if not .Values.app.secrets.openai_api_key }}
{{- fail "OpenAI API key is required" }}
{{- end }}
{{- end }}

{{/*
Generate OpenTelemetry resource attributes
*/}}
{{- define "agentic-startup-studio.otelResourceAttributes" -}}
service.name={{ include "agentic-startup-studio.fullname" . }},service.version={{ .Values.app.version }},deployment.environment={{ .Values.global.environment }},k8s.namespace.name={{ .Release.Namespace }},k8s.cluster.name={{ .Values.global.clusterName | default "default" }}
{{- end }}

{{/*
Generate common environment variables
*/}}
{{- define "agentic-startup-studio.commonEnvVars" -}}
- name: ENVIRONMENT
  value: {{ .Values.global.environment | quote }}
- name: RELEASE_NAME
  value: {{ .Release.Name | quote }}
- name: RELEASE_NAMESPACE
  value: {{ .Release.Namespace | quote }}
- name: HELM_CHART_VERSION
  value: {{ .Chart.Version | quote }}
{{- end }}