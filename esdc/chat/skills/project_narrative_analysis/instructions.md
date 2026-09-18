# Project narrative analysis playbook

## Sequence

1. Start with Semantic Search using the user's issue, project, WK, and year
   filters where available.
2. Use returned project IDs or entities in SQL to verify structured values and
   the relevant report year.
3. Read project remarks and GROOVY report as evidence for narrative claims;
   preserve their labels and distinguish GROOVY active/status.
4. Summarize evidence per project and disclose ranking, filters, NULLs, and
   contradictory records.

## Evidence boundary

Ranked semantic hits are candidates, not an exhaustive project list. SQL
aggregates establish counts and totals. Do not infer a project-wide claim from
one passage or treat a missing narrative as proof that no issue exists.

## Forbidden shortcuts

- Do not replace semantic retrieval with a broad unfiltered SQL scan.
- Do not invent project remarks, GROOVY content, or causal explanations.
- Do not mix narrative evidence from different report years without labeling it.
