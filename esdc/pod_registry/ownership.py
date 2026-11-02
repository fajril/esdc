"""DuckDB object names owned by the canonical POD projection.

Pure shared authority for the publisher and the generic loader: the reserved
names come from the canonical contract plus the objects the publisher itself
creates or retires, so no second copy of the ownership lists exists.
"""

from __future__ import annotations

from esdc.pod_registry.contract import PodContract

SUPPORT_TABLES = (
    "r_institution",
    "r_pod_type",
    "r_pod_stage",
    "pod_document",
    "pod_registry_state",
    "pod_knowledge_state",
    "pod_read_model_snapshot",
    "pod_read_model_diagnostic",
    "_pod_identity_ledger",
    "pod_registry_legacy_identity",
)
RETIRED_OBJECTS = (
    "pod_plan",
    "pod_monitoring",
    "pod_project_economics",
    "pod_economics",
    "pod_revision",
    "pod_value_case",
    "project_pod",
)
PUBLICATION_TABLE = "pod_registry_publication"
REGISTRY_VIEW = "pod_registry"


def pod_reserved_names(contract: PodContract) -> dict[str, str]:
    """Map casefolded DuckDB names to the POD-owned object they resolve to.

    DuckDB identifiers are matched case-insensitively, so `M_POD` resolves to
    the published `m_pod` projection. Names are returned casefolded with the
    canonical spelling preserved for error messages.
    """
    names = (
        *contract.tables,
        *SUPPORT_TABLES,
        *RETIRED_OBJECTS,
        PUBLICATION_TABLE,
        REGISTRY_VIEW,
    )
    return {name.casefold(): name for name in names}
