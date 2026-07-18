"""Table/grid configuration for the POD portal."""

TABLE_CONFIGS: dict[str, dict] = {
    "m_pod": {
        "title": "PODs",
        "pk": ["id"],
        "columns": [
            {"field": "id", "title": "ID (ITB)", "editableOnNew": True, "required": True},
            {"field": "approval_date", "title": "Approval Date", "editableOnNew": True, "editor": "date", "required": True},
            {"field": "institution_code", "title": "Institution", "editableOnNew": True, "ref": "institutions", "required": True},
            {"field": "pod_type_code", "title": "POD Type", "editableOnNew": True, "ref": "pod_types", "required": True},
            {"field": "rev_num", "title": "Rev", "editableOnNew": True, "required": True},
            {"field": "pod_name", "title": "Name", "editable": True, "required": True},
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
            {"field": "pod_id", "title": "POD", "editableOnNew": True, "ref": "pods", "required": True},
            {"field": "project_id", "title": "Project ID", "editableOnNew": True, "autocomplete": "/api/projects", "required": True},
        ],
    },
    "pod_document": {
        "title": "Document Links",
        "pk": ["pod_id", "doc_id"],
        "columns": [
            {"field": "pod_id", "title": "POD", "editableOnNew": True, "ref": "pods", "required": True},
            {"field": "doc_id", "title": "Document", "editableOnNew": True, "autocomplete": True, "autocompleteRef": "documents", "required": True},
        ],
    },
    "pod_revision": {
        "title": "Revisions",
        "pk": ["successor_id", "predecessor_id"],
        "columns": [
            {"field": "successor_id", "title": "Successor POD", "editableOnNew": True, "ref": "pod_ids", "required": True},
            {"field": "predecessor_id", "title": "Predecessor POD", "editableOnNew": True, "ref": "pod_ids", "required": True},
        ],
    },
    "r_institution": {
        "title": "Institutions",
        "pk": ["code"],
        "columns": [
            {"field": "code", "title": "Code", "editableOnNew": True, "required": True},
            {"field": "institution", "title": "Institution", "editable": True, "required": True},
            {"field": "description", "title": "Description", "editable": True},
        ],
    },
    "r_pod_type": {
        "title": "POD Types",
        "pk": ["code"],
        "columns": [
            {"field": "code", "title": "Code", "editableOnNew": True, "required": True},
            {"field": "pod_type", "title": "POD Type", "editable": True, "required": True},
            {"field": "description", "title": "Description", "editable": True},
        ],
    },
}
