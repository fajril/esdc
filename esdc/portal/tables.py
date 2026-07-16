"""Table/grid configuration for the POD portal."""

TABLE_CONFIGS: dict[str, dict] = {
    "m_pod": {
        "title": "PODs",
        "pk": ["id"],
        "columns": [
            {"field": "id", "title": "ID (ITB)", "editableOnNew": True},
            {"field": "approval_date", "title": "Approval Date", "editableOnNew": True, "editor": "date"},
            {"field": "institution_code", "title": "Institution", "editableOnNew": True, "ref": "institutions"},
            {"field": "pod_type_code", "title": "POD Type", "editableOnNew": True, "ref": "pod_types"},
            {"field": "rev_num", "title": "Rev", "editableOnNew": True},
            {"field": "pod_name", "title": "Name", "editable": True},
            {"field": "pod_letter_num", "title": "Letter No.", "editable": True},
            {"field": "approval_seq", "title": "Seq", "readonly": True},
            {"field": "pod_id", "title": "POD ID", "readonly": True},
            {"field": "preceded_by", "title": "Preceded By", "readonly": True},
            {"field": "superseded_by", "title": "Superseded By", "readonly": True},
        ],
    },
    "project_pod": {
        "title": "Project Links",
        "pk": ["pod_id", "project_id"],
        "columns": [
            {"field": "pod_id", "title": "POD", "editableOnNew": True, "ref": "pods"},
            {"field": "project_id", "title": "Project ID", "editableOnNew": True, "autocomplete": "/api/projects"},
        ],
    },
    "pod_revision": {
        "title": "Revisions",
        "pk": ["successor_id", "predecessor_id"],
        "columns": [
            {"field": "successor_id", "title": "Successor POD", "editableOnNew": True, "ref": "pod_ids"},
            {"field": "predecessor_id", "title": "Predecessor POD", "editableOnNew": True, "ref": "pod_ids"},
        ],
    },
    "r_institution": {
        "title": "Institutions",
        "pk": ["code"],
        "columns": [
            {"field": "code", "title": "Code", "editableOnNew": True},
            {"field": "institution", "title": "Institution", "editable": True},
            {"field": "description", "title": "Description", "editable": True},
        ],
    },
    "r_pod_type": {
        "title": "POD Types",
        "pk": ["code"],
        "columns": [
            {"field": "code", "title": "Code", "editableOnNew": True},
            {"field": "pod_type", "title": "POD Type", "editable": True},
            {"field": "description", "title": "Description", "editable": True},
        ],
    },
}
