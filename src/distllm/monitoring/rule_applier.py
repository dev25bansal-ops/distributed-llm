"""Rule applier — loads alerting rules and pushes them to Prometheus."""

import re

import requests
from loguru import logger

from distllm.errors import ConfigValidationError
from distllm.monitoring.alert_rules import (
    AlertRule,
    RecordingRule,
    get_default_alerts,
    get_default_recording_rules,
    rules_to_yaml,
)


# Minimal PromQL syntax validation — catches obvious errors without a full parser
_PROMQL_FUNC_RE = re.compile(r'[a-z_]+\(', re.IGNORECASE)
_UNBALANCED_PAREN_RE = re.compile(r'[^(]*\)[^(]*$')


def validate_promql(expr: str) -> list[str]:
    """Basic PromQL syntax validation.

    Returns a list of issues found (empty if expression looks valid).
    """
    issues = []
    if not expr.strip():
        issues.append("Expression is empty")
        return issues

    # Check balanced parentheses
    if expr.count('(') != expr.count(')'):
        issues.append(f"Unbalanced parentheses: {expr.count('(')} open, {expr.count(')')} close")

    # Check for unclosed quotes
    for quote in ('"', "'"):
        if expr.count(quote) % 2 != 0:
            issues.append(f"Unmatched quote: {quote}")

    return issues


class RuleApplier:
    """Loads and applies alerting/recording rules to a Prometheus server."""

    def __init__(self, prometheus_url: str, rules_dir: str | None = None):
        self.prometheus_url = prometheus_url.rstrip("/")
        self.rules_dir = rules_dir

    def load_and_validate(
        self,
        rule_file: str | None = None,
        use_defaults: bool = True,
    ) -> tuple:
        """Load rules from file or use defaults.

        Args:
            rule_file: Path to YAML rule file. If None, use defaults.
            use_defaults: Whether to include default rules.

        Returns:
            Tuple of (alerts, recording_rules, yaml_content).
        """
        import yaml

        alerts: list[AlertRule] = []
        recording: list[RecordingRule] = []

        if rule_file:
            with open(rule_file, "r") as f:
                data = yaml.safe_load(f)

            for group in data.get("groups", []):
                for rule in group.get("rules", []):
                    if "alert" in rule:
                        alerts.append(AlertRule(
                            alert=rule["alert"],
                            expr=rule["expr"],
                            for_duration=rule.get("for", "1m"),
                            labels=rule.get("labels", {}),
                            annotations=rule.get("annotations", {}),
                        ))
                    elif "record" in rule:
                        recording.append(RecordingRule(
                            record=rule["record"],
                            expr=rule["expr"],
                            labels=rule.get("labels", {}),
                        ))
            logger.info(f"Loaded {len(alerts)} alerts and {len(recording)} recording rules from {rule_file}")

        if use_defaults and not rule_file:
            alerts = get_default_alerts()
            recording = get_default_recording_rules()
            logger.info(f"Using default rules: {len(alerts)} alerts, {len(recording)} recording rules")

        # Validate all expressions
        all_issues = []
        for rule in alerts:
            issues = validate_promql(rule.expr)
            for issue in issues:
                all_issues.append(f"Alert '{rule.alert}': {issue}")
        for rule in recording:
            issues = validate_promql(rule.expr)
            for issue in issues:
                all_issues.append(f"Recording '{rule.record}': {issue}")

        if all_issues:
            raise ConfigValidationError("alert_rules", "\n".join(all_issues))

        yaml_content = rules_to_yaml(alerts, recording)
        return alerts, recording, yaml_content

    def apply_rules(
        self,
        rule_file: str | None = None,
        use_defaults: bool = True,
    ) -> bool:
        """Load rules and push to Prometheus via HTTP API.

        Args:
            rule_file: Path to YAML rule file.
            use_defaults: Whether to use default rules if no file provided.

        Returns:
            True if rules were applied successfully.
        """
        alerts, recording, yaml_content = self.load_and_validate(rule_file, use_defaults)

        # Write rule file to disk before triggering Prometheus reload
        if self.rules_dir is not None:
            import os
            os.makedirs(self.rules_dir, exist_ok=True)
            dest = os.path.join(self.rules_dir, "distllm_rules.yml")
            with open(dest, "w") as f:
                f.write(yaml_content)
            logger.info(f"Wrote rule file to {dest}")

        # Prometheus hot-reload endpoint
        url = f"{self.prometheus_url}/-/reload"
        try:
            response = requests.post(url, timeout=10)
            if response.status_code == 200:
                logger.info("Prometheus rules reloaded successfully")
                return True
            else:
                logger.warning(
                    f"Prometheus reload returned {response.status_code}: {response.text}"
                )
                return False
        except requests.RequestException as e:
            logger.warning(f"Could not reach Prometheus at {url}: {e}")
            return False

    def get_rules_yaml(
        self,
        rule_file: str | None = None,
        use_defaults: bool = True,
    ) -> str:
        """Return the rules as YAML string (for config generation / dry-run)."""
        _, _, yaml_content = self.load_and_validate(rule_file, use_defaults)
        return yaml_content

    def generate_configmap_yaml(
        self,
        name: str = "distllm-alert-rules",
        namespace: str = "monitoring",
        rule_file: str | None = None,
        use_defaults: bool = True,
    ) -> str:
        """Generate a Kubernetes ConfigMap YAML for Prometheus rules.

        Args:
            name: ConfigMap name.
            namespace: Kubernetes namespace.
            rule_file: Optional path to custom rule file.
            use_defaults: Whether to include default rules.

        Returns:
            Kubernetes ConfigMap YAML string.
        """
        import yaml

        _, _, rules_yaml = self.load_and_validate(rule_file, use_defaults)

        configmap = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": "distributed-llm",
                    "component": "prometheus-rules",
                },
            },
            "data": {
                "distllm-alerts.yaml": rules_yaml,
            },
        }

        return yaml.dump(configmap, default_flow_style=False, sort_keys=False)

    def deploy_configmap(
        self,
        name: str = "distllm-alert-rules",
        namespace: str = "monitoring",
        rule_file: str | None = None,
        use_defaults: bool = True,
    ) -> bool:
        """Deploy alert rules as a Kubernetes ConfigMap.

        Uses the Kubernetes API to create/update the ConfigMap,
        then triggers a Prometheus reload.

        Args:
            name: ConfigMap name.
            namespace: Kubernetes namespace.
            rule_file: Optional path to custom rule file.
            use_defaults: Whether to include default rules.

        Returns:
            True if ConfigMap was created/updated successfully.
        """
        import subprocess
        import tempfile

        configmap_yaml = self.generate_configmap_yaml(name, namespace, rule_file, use_defaults)

        # Write to temp file and apply with kubectl
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(configmap_yaml)
            f.flush()
            try:
                result = subprocess.run(
                    ["kubectl", "apply", "-f", f.name],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    logger.error(f"kubectl apply failed: {result.stderr}")
                    return False
                logger.info(f"ConfigMap {name} deployed to {namespace}")
            except FileNotFoundError:
                logger.error("kubectl not found — cannot deploy ConfigMap")
                return False
            except subprocess.TimeoutExpired:
                logger.error("kubectl apply timed out")
                return False

        # Trigger Prometheus reload
        return self.apply_rules(rule_file, use_defaults)

    def write_rules_file(
        self,
        output_path: str,
        rule_file: str | None = None,
        use_defaults: bool = True,
    ) -> bool:
        """Write rules to a Prometheus-compatible YAML file.

        Args:
            output_path: Path to write the rules file.
            rule_file: Optional input rule file.
            use_defaults: Whether to include default rules.

        Returns:
            True if file was written successfully.
        """
        try:
            yaml_content = self.get_rules_yaml(rule_file, use_defaults)
            with open(output_path, "w") as f:
                f.write(yaml_content)
            logger.info(f"Rules written to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write rules: {e}")
            return False
