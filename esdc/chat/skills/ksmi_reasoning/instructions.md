# KSMI reasoning playbook

## Sequence

1. Call Knowledge Traversal first for the requested KSMI topic and explicit
   level or entity.
2. Treat the returned knowledge as authoritative for definitions, transitions,
   formulas, and labels. Quote or summarize the returned section label.
3. If the user asks for current project, field, or WK records, use SQL only
   after the knowledge lookup and keep the database facts separate.
4. Report uncertainty or missing retrieval evidence instead of filling gaps.

## Evidence boundary

GROOVY report is content; GROOVY active is status. A retrieved definition does
not prove a particular record's current status. Never turn a semantic hit into
an exhaustive record list.

## Forbidden shortcuts

- Do not invent a transition, maturity level, formula, WAP limit, or exception.
- Do not skip the required knowledge lookup because a level token looks clear.
- Do not use SQL to answer a KSMI definition question.
