"""Problem cluster definitions for ESDC project remarks analysis.

This module contains the hierarchical classification system for oil & gas upstream
problems, enabling extraction and analysis of problem references from project remarks.
"""

from typing import Dict, List, Optional
import re


# Problem cluster code pattern: X.X.X (e.g., 1.1.1, 2.3, 3.1.2)
PROBLEM_CLUSTER_CODE_PATTERN = re.compile(
    r"\b(\d(?:\.\d+){1,2})\b"  # Matches 1.1.1, 2.3, 3.1.2, etc.
)

# Pattern to match "Problem Cluster: " prefix with variations
PROBLEM_CLUSTER_PREFIX_PATTERNS = [
    re.compile(
        r"Problem Cluster[:\s]+(\d(?:\.\d+){1,2})\.?\s*([^.\n]+)?", re.IGNORECASE
    ),
    re.compile(r"Kendala[:\s]+(\d(?:\.\d+){1,2})", re.IGNORECASE),
    re.compile(r"Cluster[:\s]+(\d(?:\.\d+){1,2})", re.IGNORECASE),
    re.compile(r"Masalah[:\s]+(\d(?:\.\d+){1,2})", re.IGNORECASE),
]


PROBLEM_CLUSTERS: Dict[str, Dict] = {
    # Technical > Subsurface
    "1.1.1": {
        "category": "Technical > Subsurface",
        "name": "Subsurface Uncertainty",
        "definition": "Ketidakpastian dari reservoir yang berdampak pada penentuan atau perhitungan volume maupun recovery factor",
        "examples": [
            "Fluid Contact tidak terdefinisi dengan jelas",
            "sebaran reservoir properties tidak dapat ditentukan dengan baik karena berada di carbonate",
        ],
        "keywords": [
            "fluid contact",
            "reservoir properties",
            "carbonate",
            "uncertainty",
            "tidak terdefinisi",
        ],
    },
    "1.1.2": {
        "category": "Technical > Subsurface",
        "name": "Data Availability",
        "definition": "Tidak adanya ketersediaan data yang cukup untuk melakukan evaluasi lebih lanjut",
        "examples": [
            "Data seismik 3D tidak tersedia",
            "Data Well Test tidak tersedia",
            "Data core tidak tersedia",
        ],
        "keywords": ["data", "tidak tersedia", "seismik", "well test", "core"],
    },
    "1.1.3": {
        "category": "Technical > Subsurface",
        "name": "Recovery Technology",
        "definition": "Belum tersedianya teknologi yang mampu mengalirkan fluida dari reservoir ke permukaan",
        "examples": [
            "Belum terdapat teknologi fabrikasi surfactant yang tersedia untuk keperluan EOR",
            "Belum tersedia teknologi multi stage horizontal well untuk pengembangan shale oil",
        ],
        "keywords": [
            "teknologi",
            "tidak tersedia",
            "EOR",
            "surfactant",
            "horizontal well",
        ],
    },
    "1.1.4": {
        "category": "Technical > Subsurface",
        "name": "Well Performance",
        "definition": "Kendala performa sumur terkait karakteristik reservoir",
        "examples": [
            "Laju alir rendah akibat permeabilitas rendah",
            "Skin factor tinggi",
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
            "Low Quality Reservoir yang memerlukan hydraulic fracturing",
        ],
        "keywords": [
            "workover",
            "well service",
            "acidizing",
            "hydraulic fracturing",
            "skin",
        ],
    },
    "1.2.2": {
        "category": "Technical > Non Subsurface",
        "name": "Production Facilities",
        "definition": "Kendala produksi yang memerlukan perubahan atau penambahan fasilitas produksi",
        "examples": [
            "Fasilitas produksi yang ada belum dapat menangani kadar CO2 yang tinggi",
            "Proyek memerlukan water treatment facility untuk dapat berproduksi",
        ],
        "keywords": [
            "fasilitas",
            "facility",
            "treatment",
            "CO2",
            "production facilities",
        ],
    },
    "1.2.3": {
        "category": "Technical > Non Subsurface",
        "name": "Flow Assurance",
        "definition": "Kendala aliran fluida dalam jaringan pipa yang memerlukan penanganan khusus namun tidak memerlukan penambahan fasilitas produksi baru atau perubahan fasilitas produksi yang signifikan",
        "examples": [
            "Implementasi heater untuk mencegah pembentukan hydrate atau wax",
            "Penambahan solvent untuk meminimalisir dampak dari H2S",
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
        "keywords": [
            "marginal",
            "fiscal term",
            "perpajakan",
            "ekonomi",
            "tidak menarik",
        ],
    },
    "2.2": {
        "category": "Economics",
        "name": "Uneconomic",
        "definition": "Kondisi keekonomian yang secara proyek sudah memiliki indikator keekonomian yang negatif sebelum diterapkan fiscal term",
        "examples": [
            "Proyek memiliki sumber daya yang sangat kecil sehingga NPV < 0",
            "Proyek memerlukan investasi yang sangat besar sehingga NPV < 0",
        ],
        "keywords": [
            "uneconomic",
            "NPV",
            "negatif",
            "tidak ekonomis",
            "investasi besar",
        ],
    },
    "2.3": {
        "category": "Economics",
        "name": "Portfolio Priority",
        "definition": "Kondisi proyek yang belum mendapatkan persetujuan investasi dari stakeholder internal kontraktor akibat evaluasi perbandingan portfolio proyek dengan proyek lain yang dikelola kontraktor",
        "examples": [
            "Proyek LNG belum mendapatkan pinjaman dari lender",
            "Portfolio di negara lain lebih menjanjikan dibanding proyek yang akan dikerjakan",
        ],
        "keywords": ["portfolio", "prioritas", "pinjaman", "lender", "investasi"],
    },
    "2.4": {
        "category": "Economics",
        "name": "Market Availability",
        "definition": "Proyek belum menemukan market yang dapat menyerap produksi",
        "examples": ["Tidak tersedia buyer"],
        "keywords": ["market", "buyer", "tidak tersedia", "penyerapan"],
    },
    "2.5": {
        "category": "Economics",
        "name": "Transportation Facilities",
        "definition": "Hasil produksi tidak dapat diserap oleh market akibat tidak ada fasilitas transmisi yang mengantarkan ke lokasi pembeli",
        "examples": ["Lokasi buyer terlalu jauh", "Tidak ada pipeline ke lokasi buyer"],
        "keywords": ["transportasi", "transmisi", "pipeline", "lokasi", "jauh"],
    },
    # Legal and Regulations > Law and Regulations
    "3.1.1": {
        "category": "Legal > Law and Regulations",
        "name": "Regulations",
        "definition": "Proyek tidak dapat berproduksi karena terkendala aturan atau regulasi yang ada, baik dari sisi pemerintah pusat maupun pemerintah daerah",
        "examples": [
            "Proyek berada pada area konservasi",
            "Proyek berada di area tumpang tindih lahan dengan kegiatan pertambangan",
        ],
        "keywords": [
            "regulasi",
            "aturan",
            "konservasi",
            "tumpang tindih",
            "pemerintah",
        ],
    },
    "3.1.2": {
        "category": "Legal > Law and Regulations",
        "name": "AMDAL",
        "definition": "Proyek belum memperoleh izin AMDAL",
        "examples": ["Proyek tidak kunjung mendapatkan persetujuan AMDAL"],
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
            "Harga tanah yang sudah diluar kewajaran",
        ],
        "keywords": ["land acquisition", "pengadaan lahan", "penolakan", "harga tanah"],
    },
    "4.2": {
        "category": "Social and Environment",
        "name": "Social Problems",
        "definition": "Proyek mengalami kendala produksi akibat adanya gejolak sosial di area operasi",
        "examples": [
            "Proyek berada di area konflik",
            "Masyarakat menolak kegiatan hulu migas",
        ],
        "keywords": ["social", "masyarakat", "konflik", "penolakan", "gejolak sosial"],
    },
    "4.3": {
        "category": "Social and Environment",
        "name": "Local Customs",
        "definition": "Proyek mengalami kendala produksi akibat dari dinamika kondisi sosial masyarakat",
        "examples": [
            "Proyek berada di tanah adat",
            "Proyek memerlukan persetujuan dari kepala adat",
        ],
        "keywords": [
            "customs",
            "adat",
            "tanah adat",
            "kepala adat",
            "persetujuan adat",
        ],
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


def search_problem_clusters(query: str, limit: int = 3) -> List[Dict]:
    """
    Search problem clusters by partial name/code match.

    Supports fuzzy matching on cluster names, codes, and keywords.
    Useful for answering questions like "apa arti subsurface uncertainty?"

    Args:
        query: Search term (e.g., "subsurface", "1.1.1", "data availability")
        limit: Maximum results to return (default: 3)

    Returns:
        List of matching clusters sorted by relevance score

    Examples:
        >>> search_problem_clusters("subsurface")
        [{'code': '1.1.1', 'name': 'Subsurface Uncertainty', ...}]

        >>> search_problem_clusters("1.1.1")  # Exact code match
        [{'code': '1.1.1', 'name': 'Subsurface Uncertainty', ...}]
    """
    query_lower = query.lower().strip()
    matches = []

    for code, info in PROBLEM_CLUSTERS.items():
        score = 0

        # Exact code match (highest priority)
        if query == code:
            score = 100
        # Partial code match
        elif query in code:
            score = 90

        # Name contains query
        if query_lower in info["name"].lower():
            score += 50

        # Keywords match
        if any(query_lower in kw.lower() for kw in info["keywords"]):
            score += 30

        if score > 0:
            matches.append({**info, "code": code, "match_score": score})

    return sorted(matches, key=lambda x: x["match_score"], reverse=True)[:limit]


def get_cluster_explanation(cluster_code: str) -> str:
    """
    Get formatted explanation for a problem cluster.

    Returns a human-readable string with definition and examples,
    suitable for displaying to users.

    Args:
        cluster_code: Problem cluster code (e.g., "1.1.1")

    Returns:
        Formatted explanation string with definition and examples

    Examples:
        >>> print(get_cluster_explanation("1.1.1"))
        **Subsurface Uncertainty** (1.1.1)
        Category: Technical > Subsurface

        Definition:
        Ketidakpastian dari reservoir yang berdampak pada penentuan...

        Examples:
        - Fluid Contact tidak terdefinisi dengan jelas
        - sebaran reservoir properties tidak dapat ditentukan...
    """
    cluster = PROBLEM_CLUSTERS.get(cluster_code)
    if not cluster:
        return f"Problem cluster '{cluster_code}' not found."

    examples_text = "\n".join(f"- {ex}" for ex in cluster["examples"])

    return f"""**{cluster["name"]}** ({cluster_code})
Category: {cluster["category"]}

Definition:
{cluster["definition"]}

Examples:
{examples_text}"""
