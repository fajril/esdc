# Spatial analysis playbook

## Sequence

1. Resolve the named field, WK, or coordinate with Spatial Resolver before SQL.
2. Preserve the requested radius, distance unit, geometry type, and WK scope.
3. Use returned entity IDs or geometry constraints in SQL; report the source
   geometry and the rows actually returned.
4. Explain when geometry is missing, approximate, or derived, and distinguish
   a spatial candidate set from an exhaustive inventory.

## Evidence boundary

`field_geom` and other geometry values are spatial evidence, not narrative
labels. Keep geometry-derived results separate from authoritative resource
metrics and disclose coordinate or distance assumptions.

## Forbidden shortcuts

- Do not approximate a radius with name matching.
- Do not drop WK scope or units while translating a spatial request to SQL.
- Do not claim that omitted geometry proves an entity is outside the area.
