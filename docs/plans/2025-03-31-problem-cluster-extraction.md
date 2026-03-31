# Problem Cluster Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract problem cluster references (e.g., "1.1.1 Subsurface Uncertainty") from project_remarks, vol_remarks, and frcast_remarks columns to enable AI analysis of project issues.

**Architecture:** Create a new `problems.py` module containing cluster definitions and extraction logic. Support multiple clusters per project with flexible pattern matching for various text formats. Integrate extraction into project data display.

**Tech Stack:** Python, Regex, Existing domain_knowledge package structure

---

## Task 1: Create `problems.py` Module with Cluster Definitions

**Files:**
- Create: `esdc/chat/domain_knowledge/problems.py`

**Step 1: Write the module structure**

```python
"""Problem cluster definitions for ESDC project remarks analysis.

This module contains the hierarchical classification system for oil & gas upstream
problems, enabling extraction and analysis of problem references from project remarks.
"""

from typing import Dict, List, Optional
import re


# Problem cluster code pattern: X.X.X (e.g., 1.1.1, 2.3, 3.1.2)
PROBLEM_CLUSTER_CODE_PATTERN = re.compile(
    r'\b(\d(?:\.\d+){1,2})\b'  # Matches 1.1.1, 2.3, 3.1.2, etc.
)

# Pattern to match "Problem Cluster: " prefix with variations
PROBLEM_CLUSTER_PREFIX_PATTERNS = [
    re.compile(r'Problem Cluster[:\s]+(\d(?:\.\d+){1,2})\.?\s*([^.\n]+)?', re.IGNORECASE),
    re.compile(r'Kendala[:\s]+(\d(?:\.\d+){1,2})', re.IGNORECASE),
    re.compile(r'Cluster[:\s]+(\d(?:\.\d+){1,2})', re.IGNORECASE),
    re.compile(r'Masalah[:\s]+(\d(?:\.\d+){1,2})', re.IGNORECASE),
]


PROBLEM_CLUSTERS: Dict[str, Dict] = {
    # Technical > Subsurface
    "1.1.1": {
        "category": "Technical > Subsurface",
        "name": "Subsurface Uncertainty",
        "definition": "Ketidakpastian dari reservoir yang berdampak pada penentuan atau perhitungan volume maupun recovery factor",
        "examples": [
            "Fluid Contact tidak terdefinisi dengan jelas",
            "sebaran reservoir properties tidak dapat ditentukan dengan baik karena berada di carbonate"
        ],
        "keywords": ["fluid contact", "reservoir properties", "carbonate", "uncertainty", "tidak terdefinisi"],
    },
    "1.1.2": {
        "category": "Technical > Subsurface",
        "name": "Data Availability",
        "definition": "Tidak adanya ketersediaan data yang cukup untuk melakukan evaluasi lebih lanjut",
        "examples": [
            "Data seismik 3D tidak tersedia",
            "Data Well Test tidak tersedia",
            "Data core tidak tersedia"
        ],
        "keywords": ["data", "tidak tersedia", "seismik", "well test", "core"],
    },
    "1.1.3": {
        "category": "Technical > Subsurface",
        "name": "Recovery Technology",
        "definition": "Belum tersedianya teknologi yang mampu mengalirkan fluida dari reservoir ke permukaan",
        "examples": [
            "Belum terdapat teknologi fabrikasi surfactant yang tersedia untuk keperluan EOR",
            "Belum tersedia teknologi multi stage horizontal well untuk pengembangan shale oil"
        ],
        "keywords": ["teknologi", "tidak tersedia", "EOR", "surfactant", "horizontal well"],
    },
    "1.1.4": {
        "category": "Technical > Subsurface",
        "name": "Well Performance",
        "definition": "Kendala performa sumur terkait karakteristik reservoir",
        "examples": [
            "Laju alir rendah akibat permeabilitas rendah",
            "Skin factor tinggi"
        ],
        "keywords": ["well performance", "laju alir", "permeabilitas", "skin factor"],
    },
    # Technical > Non Subsurface
    "1.2.1": {
        "category": "Technical > Non Subsurface",
        "name": "Well Interventions",
        "definition": "Kendala di sumur yang memerlukan kegiatan workover atau well service",
        "examples": [
            "Sumur mengalami skin problem sehingga memerlukan acidizing",
            "Low Quality Reservoir yang memerlukan hydraulic fracturing"
        ],
        "keywords": ["workover", "well service", "acidizing", "hydraulic fracturing", "skin"],
    },
    "1.2.2": {
        "category": "Technical > Non Subsurface",
        "name": "Production Facilities",
        "definition": "Kendala produksi yang memerlukan perubahan atau penambahan fasilitas produksi",
        "examples": [
            "Fasilitas produksi yang ada belum dapat menangani kadar CO2 yang tinggi",
            "Proyek memerlukan water treatment facility untuk dapat berproduksi"
        ],
        "keywords": ["fasilitas", "facility", "treatment", "CO2", "production facilities"],
    },
    "1.2.3": {
        "category": "Technical > Non Subsurface",
        "name": "Flow Assurance",
        "definition": "Kendala aliran fluida dalam jaringan pipa yang memerlukan penanganan khusus namun tidak memerlukan penambahan fasilitas produksi baru atau perubahan fasilitas produksi yang signifikan",
        "examples": [
            "Implementasi heater untuk mencegah pembentukan hydrate atau wax",
            "Penambahan solvent untuk meminimalisir dampak dari H2S"
        ],
        "keywords": ["flow assurance", "hydrate", "wax", "heater", "H2S", "solvent"],
    },
    # Economics
    "2.1": {
        "category": "Economics",
        "name": "Marginal",
        "definition": "Kondisi keekonomian yang secara proyek masih positif namun setelah diterapkan fiscal term menjadi tidak menarik bagi kontraktor",
        "examples": [
            "Proyek terkena skema perpajakan yang membuat indikator keekonomian tidak menarik bagi kontraktor"
        ],
        "keywords": ["marginal", "fiscal term", "perpajakan", "ekonomi", "tidak menarik"],
    },
    "2.2": {
        "category": "Economics",
        "name": "Uneconomic",
        "definition": "Kondisi keekonomian yang secara proyek sudah memiliki indikator keekonomian yang negatif sebelum diterapkan fiscal term",
        "examples": [
            "Proyek memiliki sumber daya yang sangat kecil sehingga NPV < 0",
            "Proyek memerlukan investasi yang sangat besar sehingga NPV < 0"
        ],
        "keywords": ["uneconomic", "NPV", "negatif", "tidak ekonomis", "investasi besar"],
    },
    "2.3": {
        "category": "Economics",
        "name": "Portfolio Priority",
        "definition": "Kondisi proyek yang belum mendapatkan persetujuan investasi dari stakeholder internal kontraktor akibat evaluasi perbandingan portfolio proyek dengan proyek lain yang dikelola kontraktor",
        "examples": [
            "Proyek LNG belum mendapatkan pinjaman dari lender",
            "Portfolio di negara lain lebih menjanjikan dibanding proyek yang akan dikerjakan"
        ],
        "keywords": ["portfolio", "prioritas", "pinjaman", "lender", "investasi"],
    },
    "2.4": {
        "category": "Economics",
        "name": "Market Availability",
        "definition": "Proyek belum menemukan market yang dapat menyerap produksi",
        "examples": [
            "Tidak tersedia buyer"
        ],
        "keywords": ["market", "buyer", "tidak tersedia", "penyerapan"],
    },
    "2.5": {
        "category": "Economics",
        "name": "Transportation Facilities",
        "definition": "Hasil produksi tidak dapat diserap oleh market akibat tidak ada fasilitas transmisi yang mengantarkan ke lokasi pembeli",
        "examples": [
            "Lokasi buyer terlalu jauh",
            "Tidak ada pipeline ke lokasi buyer"
        ],
        "keywords": ["transportasi", "transmisi", "pipeline", "lokasi", "jauh"],
    },
    # Legal and Regulations > Law and Regulations
    "3.1.1": {
        "category": "Legal > Law and Regulations",
        "name": "Regulations",
        "definition": "Proyek tidak dapat berproduksi karena terkendala aturan atau regulasi yang ada, baik dari sisi pemerintah pusat maupun pemerintah daerah",
        "examples": [
            "Proyek berada pada area konservasi",
            "Proyek berada di area tumpang tindih lahan dengan kegiatan pertambangan"
        ],
        "keywords": ["regulasi", "aturan", "konservasi", "tumpang tindih", "pemerintah"],
    },
    "3.1.2": {
        "category": "Legal > Law and Regulations",
        "name": "AMDAL",
        "definition": "Proyek belum memperoleh izin AMDAL",
        "examples": [
            "Proyek tidak kunjung mendapatkan persetujuan AMDAL"
        ],
        "keywords": ["AMDAL", "izin lingkungan", "persetujuan"],
    },
    "3.1.3": {
        "category": "Legal > Law and Regulations",
        "name": "Permit",
        "definition": "Proyek belum mendapatkan segala perizinan yang diperlukan diluar AMDAL",
        "examples": [
            "Proyek belum mendapatkan izin untuk beroperasi karena berada di area latihan Angkatan Laut"
        ],
        "keywords": ["permit", "izin", "beroperasi", "latihan militer"],
    },
    "3.1.4": {
        "category": "Legal > Law and Regulations",
        "name": "Geopolitics",
        "definition": "Proyek berada pada area yang mengalami perselisihan antar negara atau adanya hambatan pengembangan akibat faktor kepentingan negara tertentu",
        "examples": [
            "Proyek berada pada perbatasan negara Indonesia dan Malaysia yang batas negaranya belum disepakati kedua belah pihak"
        ],
        "keywords": ["geopolitik", "perbatasan", "perselisihan", "negara"],
    },
    # Legal > T&C Contracts
    "3.2.1": {
        "category": "Legal > T&C Contracts",
        "name": "PSC",
        "definition": "Proyek mengalami kendala produksi akibat adanya batasan dalam PSC",
        "examples": [
            "Proyek tidak dapat berproduksi karena durasi produksi akan sangat singkat karena tahun onstream dekat dengan batas PSC"
        ],
        "keywords": ["PSC", "batasan kontrak", "durasi", "kontrak"],
    },
    "3.2.2": {
        "category": "Legal > T&C Contracts",
        "name": "Sales Agreement",
        "definition": "Proyek mengalami kendala produksi akibat adanya batasan dalam Sales Agreement",
        "examples": [
            "Proyek tidak dapat berproduksi karena adanya ketidaksepakatan atas T&C dalam Sales Agreement yang ada"
        ],
        "keywords": ["sales agreement", "ketidaksepakatan", "T&C", "kontrak jual beli"],
    },
    # Social and Environment
    "4.1": {
        "category": "Social and Environment",
        "name": "Land Acquisition",
        "definition": "Proyek mengalami kendala produksi akibat masalah dalam pengadaan lahan",
        "examples": [
            "Terdapat penolakan dari masyarakat yang terdampak",
            "Harga tanah yang sudah diluar kewajaran"
        ],
        "keywords": ["land acquisition", "pengadaan lahan", "penolakan", "harga tanah"],
    },
    "4.2": {
        "category": "Social and Environment",
        "name": "Social Problems",
        "definition": "Proyek mengalami kendala produksi akibat adanya gejolak sosial di area operasi",
        "examples": [
            "Proyek berada di area konflik",
            "Masyarakat menolak kegiatan hulu migas"
        ],
        "keywords": ["social", "masyarakat", "konflik", "penolakan", "gejolak sosial"],
    },
    "4.3": {
        "category": "Social and Environment",
        "name": "Local Customs",
        "definition": "Proyek mengalami kendala produksi akibat dari dinamika kondisi sosial masyarakat",
        "examples": [
            "Proyek berada di tanah adat",
            "Proyek memerlukan persetujuan dari kepala adat"
        ],
        "keywords": ["customs", "adat", "tanah adat", "kepala adat", "persetujuan adat"],
    },
}


def get_problem_cluster(code: str) -> Optional[Dict]:
    """
    Get problem cluster definition by code.
    
    Args:
        code: Problem cluster code (e.g., "1.1.1", "2.3")
    
    Returns:
        Dict with cluster info or None if not found
    """
    return PROBLEM_CLUSTERS.get(code)


def get_all_problem_clusters() -> Dict[str, Dict]:
    """
    Get all problem cluster definitions.
    
    Returns:
        Dict mapping cluster codes to their definitions
    """
    return PROBLEM_CLUSTERS.copy()


def get_clusters_by_category(category: str) -> Dict[str, Dict]:
    """
    Get all problem clusters in a specific category.
    
    Args:
        category: Category name (e.g., "Technical > Subsurface", "Economics")
    
    Returns:
        Dict of matching clusters
    """
    return {
        code: info 
        for code, info in PROBLEM_CLUSTERS.items() 
        if category.lower() in info["category"].lower()
    }
```

**Step 2: Create the file**

Run:
```bash
cat > esdc/chat/domain_knowledge/problems.py << 'EOF'
[Paste content from Step 1]
EOF
```

**Step 3: Verify file creation**

Run:
```bash
ls -la esdc/chat/domain_knowledge/problems.py
head -20 esdc/chat/domain_knowledge/problems.py
```

Expected:
- File exists
- Contains PROBLEM_CLUSTERS dict with 20 entries

**Step 4: Commit**

```bash
git add esdc/chat/domain_knowledge/problems.py
git commit -m "feat: add problem cluster definitions with all 20 categories"
```

---

## Task 2: Add Extraction Functions to `problems.py`

**Files:**
- Modify: `esdc/chat/domain_knowledge/problems.py` (append to existing file)

**Step 1: Add extraction functions**

Append to `problems.py`:

```python

def extract_problem_clusters(text: str) -> List[Dict]:
    """
    Extract problem cluster references from text.
    
    Supports multiple clusters and various text formats:
    - "Problem Cluster: 1.1.1. Subsurface Uncertainty"
    - "Problem Cluster 1.1.1"
    - "Kendala: 1.1.1"
    - "Cluster: 1.1.1"
    - Standalone "1.1.1"
    
    Args:
        text: Text to analyze (e.g., project_remarks content)
    
    Returns:
        List of dicts with cluster info. Each dict contains:
        - code: Cluster code (e.g., "1.1.1")
        - name: Cluster name
        - category: Hierarchical category
        - definition: Full definition
        - extracted_text: Original text snippet containing the reference
    
    Examples:
        >>> text = "Problem Cluster: 1.1.1. Subsurface Uncertainty. Data tidak tersedia"
        >>> clusters = extract_problem_clusters(text)
        >>> clusters[0]["code"]
        '1.1.1'
    """
    if not text:
        return []
    
    clusters = []
    found_codes = set()
    
    # Strategy 1: Look for prefixed patterns (more specific)
    for pattern in PROBLEM_CLUSTER_PREFIX_PATTERNS:
        for match in pattern.finditer(text):
            code = match.group(1)
            if code in PROBLEM_CLUSTERS and code not in found_codes:
                cluster_info = PROBLEM_CLUSTERS[code].copy()
                cluster_info["code"] = code
                cluster_info["extracted_text"] = match.group(0)
                clusters.append(cluster_info)
                found_codes.add(code)
    
    # Strategy 2: Look for standalone cluster codes
    for match in PROBLEM_CLUSTER_CODE_PATTERN.finditer(text):
        code = match.group(1)
        if code in PROBLEM_CLUSTERS and code not in found_codes:
            cluster_info = PROBLEM_CLUSTERS[code].copy()
            cluster_info["code"] = code
            # Get surrounding context (50 chars before/after)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            cluster_info["extracted_text"] = text[start:end].strip()
            clusters.append(cluster_info)
            found_codes.add(code)
    
    return clusters


def extract_problem_clusters_from_project(project_data: Dict) -> List[Dict]:
    """
    Extract problem clusters from all remarks fields in a project.
    
    Checks: project_remarks, vol_remarks, frcast_remarks
    
    Args:
        project_data: Dict containing project fields
    
    Returns:
        Combined list of unique problem clusters from all remarks
    """
    all_clusters = []
    found_codes = set()
    
    remarks_fields = [
        "project_remarks",
        "vol_remarks", 
        "frcast_remarks",
    ]
    
    for field in remarks_fields:
        text = project_data.get(field, "")
        if text:
            clusters = extract_problem_clusters(text)
            for cluster in clusters:
                code = cluster["code"]
                if code not in found_codes:
                    all_clusters.append(cluster)
                    found_codes.add(code)
    
    return all_clusters


def enrich_project_with_clusters(project_data: Dict) -> Dict:
    """
    Enrich project data with extracted problem clusters.
    
    Adds a 'problem_clusters' key containing list of extracted clusters.
    Original project_data is not modified (returns new dict).
    
    Args:
        project_data: Original project data dict
    
    Returns:
        New dict with added 'problem_clusters' key
    """
    enriched = project_data.copy()
    enriched["problem_clusters"] = extract_problem_clusters_from_project(project_data)
    return enriched


def get_projects_by_cluster(projects: List[Dict], cluster_code: str) -> List[Dict]:
    """
    Filter projects that contain a specific problem cluster.
    
    Args:
        projects: List of project data dicts
        cluster_code: Problem cluster code to search for (e.g., "1.1.1")
    
    Returns:
        List of projects containing the specified cluster
    """
    matching_projects = []
    
    for project in projects:
        clusters = extract_problem_clusters_from_project(project)
        cluster_codes = [c["code"] for c in clusters]
        if cluster_code in cluster_codes:
            matching_projects.append(project)
    
    return matching_projects


def get_cluster_summary(projects: List[Dict]) -> Dict[str, int]:
    """
    Get summary of problem clusters across multiple projects.
    
    Args:
        projects: List of project data dicts
    
    Returns:
        Dict mapping cluster codes to occurrence counts
    """
    summary = {}
    
    for project in projects:
        clusters = extract_problem_clusters_from_project(project)
        for cluster in clusters:
            code = cluster["code"]
            summary[code] = summary.get(code, 0) + 1
    
    return summary


def format_cluster_for_display(cluster: Dict) -> str:
    """
    Format a problem cluster for display.
    
    Args:
        cluster: Cluster dict with code, name, category, etc.
    
    Returns:
        Formatted string for display
    """
    return f"{cluster['code']} - {cluster['name']} ({cluster['category']})"
```

**Step 2: Append to file**

Run:
```bash
cat >> esdc/chat/domain_knowledge/problems.py << 'EOF'
[Paste content from Step 1]
EOF
```

**Step 3: Verify extraction**

Create test script:
```python
# test_extraction.py
from esdc.chat.domain_knowledge.problems import extract_problem_clusters

test_cases = [
    "Problem Cluster: 1.1.1. Subsurface Uncertainty. Data tidak tersedia",
    "Kendala: 2.1 Marginal economics",
    "Cluster 3.1.1 Regulations issue",
    "Multiple: 1.1.1 and 2.3 Portfolio Priority",
    "Just standalone code 4.1",
]

for text in test_cases:
    clusters = extract_problem_clusters(text)
    print(f"Text: {text[:50]}...")
    print(f"Found: {[c['code'] for c in clusters]}\n")
```

Run:
```bash
python3 test_extraction.py
```

Expected:
- All test cases extract correct cluster codes
- Multiple clusters extracted from single text

**Step 4: Cleanup test file**

```bash
rm test_extraction.py
```

**Step 5: Commit**

```bash
git add esdc/chat/domain_knowledge/problems.py
git commit -m "feat: add problem cluster extraction functions"
```

---

## Task 3: Update `__init__.py` Exports

**Files:**
- Modify: `esdc/chat/domain_knowledge/__init__.py`

**Step 1: Add imports**

Add to imports section (after uncertainty imports):

```python
# =============================================================================
# Problems Module
# =============================================================================
from .problems import (
    PROBLEM_CLUSTERS,
    get_problem_cluster,
    get_all_problem_clusters,
    get_clusters_by_category,
    extract_problem_clusters,
    extract_problem_clusters_from_project,
    enrich_project_with_clusters,
    get_projects_by_cluster,
    get_cluster_summary,
    format_cluster_for_display,
)
```

**Step 2: Add to __all__**

Add to __all__ list:

```python
    # Problems
    "PROBLEM_CLUSTERS",
    "get_problem_cluster",
    "get_all_problem_clusters",
    "get_clusters_by_category",
    "extract_problem_clusters",
    "extract_problem_clusters_from_project",
    "enrich_project_with_clusters",
    "get_projects_by_cluster",
    "get_cluster_summary",
    "format_cluster_for_display",
```

**Step 3: Update file**

Edit: `esdc/chat/domain_knowledge/__init__.py`

**Step 4: Verify exports**

Run:
```bash
python3 -c "from esdc.chat.domain_knowledge import extract_problem_clusters; print('Export OK')"
```

Expected: `Export OK`

**Step 5: Commit**

```bash
git add esdc/chat/domain_knowledge/__init__.py
git commit -m "feat: export problem cluster functions from domain_knowledge package"
```

---

## Task 4: Write Tests

**Files:**
- Create: `tests/test_problem_clusters.py`

**Step 1: Create test file**

```python
"""Tests for problem cluster extraction functionality."""

import pytest
from esdc.chat.domain_knowledge.problems import (
    PROBLEM_CLUSTERS,
    get_problem_cluster,
    get_clusters_by_category,
    extract_problem_clusters,
    extract_problem_clusters_from_project,
    enrich_project_with_clusters,
    get_projects_by_cluster,
    get_cluster_summary,
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
        """Should have exactly 20 problem clusters."""
        assert len(PROBLEM_CLUSTERS) == 20
    
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
```

**Step 2: Create file**

Run:
```bash
cat > tests/test_problem_clusters.py << 'EOF'
[Paste test content]
EOF
```

**Step 3: Run tests**

Run:
```bash
python3 -m pytest tests/test_problem_clusters.py -v
```

Expected:
- All 15+ tests pass
- Coverage includes definitions, extraction, enrichment, queries

**Step 4: Commit**

```bash
git add tests/test_problem_clusters.py
git commit -m "test: add comprehensive tests for problem cluster extraction"
```

---

## Task 5: Create Integration Helper (Optional Enhancement)

**Files:**
- Modify: `esdc/chat/domain_knowledge/functions.py` (add convenience function)

**Step 1: Add helper function**

Append to `functions.py`:

```python
def analyze_project_problems(project_data: Dict) -> Dict[str, Any]:
    """
    Analyze problems mentioned in project remarks.
    
    Convenience wrapper that extracts clusters and provides analysis.
    
    Args:
        project_data: Project dict with remarks fields
    
    Returns:
        Dict with:
        - clusters: List of extracted problem clusters
        - count: Number of unique problems
        - categories: List of affected categories
        - summary: Human-readable summary
    """
    from .problems import extract_problem_clusters_from_project
    
    clusters = extract_problem_clusters_from_project(project_data)
    
    return {
        "clusters": clusters,
        "count": len(clusters),
        "categories": list(set(c["category"] for c in clusters)),
        "summary": ", ".join([f"{c['code']} ({c['name']})" for c in clusters]) if clusters else "No problems identified",
    }
```

**Step 2: Update __init__.py**

Add to imports and __all__:
```python
from .functions import (
    ...existing imports...,
    analyze_project_problems,
)
```

**Step 3: Commit**

```bash
git add esdc/chat/domain_knowledge/functions.py esdc/chat/domain_knowledge/__init__.py
git commit -m "feat: add analyze_project_problems helper function"
```

---

## Summary

**What was built:**
1. **problems.py** - Complete problem cluster definitions (20 clusters) with extraction logic
2. **Flexible extraction** - Handles multiple formats: "Problem Cluster:", "Kendala:", standalone codes
3. **Multiple clusters** - Supports extracting 1+ clusters from a single project
4. **Enrichment functions** - Automatically adds `problem_clusters` to project data
5. **Query helpers** - Filter projects by cluster, summarize across projects
6. **Tests** - Comprehensive test coverage

**Key features:**
- All cluster codes: 1.1.1-1.1.4, 1.2.1-1.2.3, 2.1-2.5, 3.1.1-3.1.4, 3.2.1-3.2.2, 4.1-4.3
- Extraction handles variations in text format
- Maintains original text snippet for context
- Supports filtering by category (Technical, Economics, Legal, Social)

**Ready for use:**
- Import: `from esdc.chat.domain_knowledge import extract_problem_clusters, enrich_project_with_clusters`
- Auto-extract: `enrich_project_with_clusters(project_data)` adds `problem_clusters` key
- Query: `get_projects_by_cluster(projects, "1.1.1")` finds projects with specific issues

**Plan saved to:** `docs/plans/2025-03-31-problem-cluster-extraction.md`
