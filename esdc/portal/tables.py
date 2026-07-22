"""Table/grid configuration for the POD portal."""

_M_POD_ROWS_SQL = """
SELECT m.*,
    (SELECT group_concat(predecessor_id, '; ') FROM pod_revision
      WHERE successor_id = m.pod_id) AS preceded_by,
    (SELECT group_concat(successor_id, '; ') FROM pod_revision
      WHERE predecessor_id = m.pod_id) AS superseded_by
FROM m_pod m ORDER BY m.approval_seq
"""

TABLE_CONFIGS: dict[str, dict] = {
    "m_pod": {
        "title": "PODs",
        "pk": ["id"],
        "sql": _M_POD_ROWS_SQL,
        "columns": [
            {
                "field": "id",
                "title": "ID (ITB)",
                "editableOnNew": True,
                "required": True,
            },
            {
                "field": "approval_date",
                "title": "Approval Date",
                "editableOnNew": True,
                "editor": "date",
                "required": True,
            },
            {
                "field": "institution_code",
                "title": "Institution",
                "editableOnNew": True,
                "ref": "institutions",
                "required": True,
            },
            {
                "field": "pod_type_code",
                "title": "POD Type",
                "editableOnNew": True,
                "ref": "pod_types",
                "required": True,
            },
            {
                "field": "rev_num",
                "title": "Rev",
                "editableOnNew": True,
                "required": True,
            },
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
            {
                "field": "pod_id",
                "title": "POD",
                "editableOnNew": True,
                "ref": "pods",
                "required": True,
            },
            {
                "field": "project_id",
                "title": "Project ID",
                "editableOnNew": True,
                "autocomplete": "/api/projects",
                "required": True,
            },
        ],
    },
    "documents": {
        "title": "Documents (corpus)",
        "pk": ["doc_id"],
        "height": "45vh",
        "sql": """
            SELECT d.doc_id, d.file_name, d.doc_type, d.doc_date, d.subject,
                   d.wk_name, d.field_name, d.project_name,
                   d.pod_name, d.suggested_pod_ids,
                   COUNT(pd.pod_id) AS linked_pods
            FROM documents d
            LEFT JOIN pod_document pd ON pd.doc_id = d.doc_id
            GROUP BY d.doc_id
            ORDER BY d.doc_date
        """,
        "columns": [
            {
                "field": "doc_id",
                "title": "Doc ID",
                "readonly": True,
                "headerFilter": True,
            },
            {
                "field": "file_name",
                "title": "File",
                "readonly": True,
                "headerFilter": True,
            },
            {
                "field": "doc_type",
                "title": "Type",
                "readonly": True,
                "headerFilter": True,
            },
            {
                "field": "doc_date",
                "title": "Date",
                "readonly": True,
                "headerFilter": True,
            },
            {
                "field": "subject",
                "title": "Subject",
                "readonly": True,
                "headerFilter": True,
            },
            {
                "field": "wk_name",
                "title": "Working Area",
                "editable": True,
                "entityList": True,
                "headerFilter": True,
            },
            {
                "field": "field_name",
                "title": "Field",
                "editable": True,
                "entityList": True,
                "headerFilter": True,
            },
            {
                "field": "project_name",
                "title": "Project",
                "editable": True,
                "entityList": True,
                "headerFilter": True,
            },
            {
                "field": "pod_name",
                "title": "POD Name",
                "readonly": True,
                "headerFilter": True,
            },
            {
                "field": "suggested_pod_ids",
                "title": "Suggested POD",
                "readonly": True,
                "headerFilter": True,
            },
            {
                "field": "linked_pods",
                "title": "Linked",
                "readonly": True,
                "headerFilter": True,
            },
        ],
    },
    "pod_document": {
        "title": "Document Links",
        "pk": ["pod_id", "doc_id"],
        "height": "40vh",
        "columns": [
            {
                "field": "pod_id",
                "title": "POD",
                "editableOnNew": True,
                "ref": "pods",
                "required": True,
            },
            {
                "field": "doc_id",
                "title": "Document",
                "editableOnNew": True,
                "autocomplete": True,
                "autocompleteRef": "documents",
                "required": True,
            },
        ],
    },
    "pod_revision": {
        "title": "Revisions",
        "pk": ["successor_id", "predecessor_id"],
        "columns": [
            {
                "field": "successor_id",
                "title": "Successor POD",
                "editableOnNew": True,
                "ref": "pod_ids",
                "required": True,
            },
            {
                "field": "predecessor_id",
                "title": "Predecessor POD",
                "editableOnNew": True,
                "ref": "pod_ids",
                "required": True,
            },
        ],
    },
    "r_institution": {
        "title": "Institutions",
        "pk": ["code"],
        "columns": [
            {"field": "code", "title": "Code", "editableOnNew": True, "required": True},
            {
                "field": "institution",
                "title": "Institution",
                "editable": True,
                "required": True,
            },
            {"field": "description", "title": "Description", "editable": True},
        ],
    },
    "r_pod_type": {
        "title": "POD Types",
        "pk": ["code"],
        "columns": [
            {"field": "code", "title": "Code", "editableOnNew": True, "required": True},
            {
                "field": "pod_type",
                "title": "POD Type",
                "editable": True,
                "required": True,
            },
            {"field": "description", "title": "Description", "editable": True},
        ],
    },
}
