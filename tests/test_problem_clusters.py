"""Tests for problem cluster extraction functionality."""

from esdc.chat.domain_knowledge.problems import (
    PROBLEM_CLUSTERS,
    enrich_project_with_clusters,
    extract_problem_clusters,
    extract_problem_clusters_from_project,
    get_cluster_summary,
    get_clusters_by_category,
    get_problem_cluster,
    get_projects_by_cluster,
)


class TestProblemClusterDefinitions:
    """Test problem cluster data structure."""

    def test_all_clusters_have_required_fields(self):
        """All clusters must have code, name, category, definition."""
        for code, cluster in PROBLEM_CLUSTERS.items():
            assert "name" in cluster
            assert "category" in cluster
            assert "definition" in cluster
            assert "examples" in cluster

    def test_total_cluster_count(self):
        """Should have 21 problem clusters (including Well Performance)."""
        assert len(PROBLEM_CLUSTERS) == 21

    def test_get_problem_cluster_existing(self):
        """Get existing cluster by code."""
        cluster = get_problem_cluster("1.1.1")
        assert cluster is not None
        assert cluster["name"] == "Subsurface Uncertainty"
        assert "Technical" in cluster["category"]

    def test_get_problem_cluster_nonexistent(self):
        """Get non-existent cluster returns None."""
        cluster = get_problem_cluster("9.9.9")
        assert cluster is None

    def test_get_clusters_by_category(self):
        """Filter clusters by category."""
        technical = get_clusters_by_category("Technical")
        assert len(technical) > 0
        for code, cluster in technical.items():
            assert "Technical" in cluster["category"]


class TestExtractProblemClusters:
    """Test extraction from text."""

    def test_extract_standard_format(self):
        """Extract from standard format."""
        text = "Problem Cluster: 1.1.1. Subsurface Uncertainty"
        clusters = extract_problem_clusters(text)
        assert len(clusters) == 1
        assert clusters[0]["code"] == "1.1.1"

    def test_extract_kendala_format(self):
        """Extract from Indonesian 'Kendala' format."""
        text = "Kendala: 2.1 Marginal economics"
        clusters = extract_problem_clusters(text)
        assert len(clusters) == 1
        assert clusters[0]["code"] == "2.1"

    def test_extract_multiple_clusters(self):
        """Extract multiple clusters from single text."""
        text = "Problem Cluster: 1.1.1. Subsurface dan Problem Cluster: 2.3 Portfolio"
        clusters = extract_problem_clusters(text)
        codes = [c["code"] for c in clusters]
        assert "1.1.1" in codes
        assert "2.3" in codes

    def test_extract_standalone_code(self):
        """Extract standalone cluster code."""
        text = "Proyek mengalami masalah 1.1.1 dan 2.1"
        clusters = extract_problem_clusters(text)
        codes = [c["code"] for c in clusters]
        assert "1.1.1" in codes
        assert "2.1" in codes

    def test_extract_empty_text(self):
        """Extract from empty text returns empty list."""
        clusters = extract_problem_clusters("")
        assert clusters == []

    def test_extract_no_clusters(self):
        """Extract from text without clusters returns empty list."""
        text = "Proyek berjalan lancar tanpa kendala"
        clusters = extract_problem_clusters(text)
        assert clusters == []

    def test_extract_with_remarks_fields(self):
        """Extract from project dict with remarks."""
        project = {
            "project_name": "Test Project",
            "project_remarks": "Problem Cluster: 1.1.1. Subsurface Uncertainty",
            "vol_remarks": "Data issue",
            "frcast_remarks": None,
        }
        clusters = extract_problem_clusters_from_project(project)
        assert len(clusters) == 1
        assert clusters[0]["code"] == "1.1.1"


class TestEnrichProject:
    """Test project enrichment."""

    def test_enrich_project_with_clusters(self):
        """Add problem_clusters key to project."""
        project = {
            "project_name": "Test",
            "project_remarks": "Problem Cluster: 1.1.1",
        }
        enriched = enrich_project_with_clusters(project)
        assert "problem_clusters" in enriched
        assert len(enriched["problem_clusters"]) == 1
        assert enriched["problem_clusters"][0]["code"] == "1.1.1"

    def test_enrich_does_not_modify_original(self):
        """Original project dict should not be modified."""
        project = {
            "project_name": "Test",
            "project_remarks": "Problem Cluster: 1.1.1",
        }
        original_keys = set(project.keys())
        enrich_project_with_clusters(project)
        assert set(project.keys()) == original_keys


class TestClusterQueries:
    """Test querying projects by clusters."""

    def test_get_projects_by_cluster(self):
        """Filter projects by cluster code."""
        projects = [
            {"project_name": "A", "project_remarks": "Problem Cluster: 1.1.1"},
            {"project_name": "B", "project_remarks": "Problem Cluster: 2.1"},
            {"project_name": "C", "project_remarks": "Problem Cluster: 1.1.1"},
        ]
        matching = get_projects_by_cluster(projects, "1.1.1")
        assert len(matching) == 2
        assert all(p["project_name"] in ["A", "C"] for p in matching)

    def test_get_cluster_summary(self):
        """Summarize clusters across projects."""
        projects = [
            {"project_remarks": "1.1.1"},
            {"project_remarks": "1.1.1"},
            {"project_remarks": "2.1"},
        ]
        summary = get_cluster_summary(projects)
        assert summary["1.1.1"] == 2
        assert summary["2.1"] == 1
