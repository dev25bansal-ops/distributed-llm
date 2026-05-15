"""Kopf operator main entry point.

Run with: kopf run src/distllm/operator/main.py
"""

import kopf

from distllm.operator.controllers.cluster_controller import (
    create_cluster,
    update_cluster,
    delete_cluster,
    reconcile_cluster,
)
from distllm.operator.controllers.nodepool_controller import (
    create_nodepool,
    update_nodepool,
    delete_nodepool,
)


@kopf.on.startup()
def setup(settings, **kwargs):
    settings.persistence.finalizer = "distllm.zeroroute.ai/finalizer"
    settings.persistence.progress_storage = kopf.AnnotationsProgressStorage()
    settings.admission.server = kopf.AdmissionServer()


if __name__ == "__main__":
    kopf.run()
