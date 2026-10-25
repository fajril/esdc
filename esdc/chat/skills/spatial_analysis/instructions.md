# Spatial analysis playbook

## Sequence

1. Resolve the named field, WK, or coordinate with Spatial Resolver before SQL.
2. Preserve the requested radius, distance unit, geometry type, and WK scope.
3. Use returned entity IDs or geometry constraints in SQL; report the geometry
   source and the rows actually returned.
4. Explain when geometry is missing, approximate, or derived, and distinguish
   a spatial candidate set from an exhaustive inventory.

## SQLite-backed visualization overlays

When a request includes a map or spatial overlay, use the active DuckDB
connection exposed through the existing `DB_PATH` contract. Query
`spatial_layer_catalog` first and use only overlay tables whose names and
metadata are published there (including `user_spatial_<id>` tables). Do not
invent table names or read an unpublished table.

Use authoritative geometry resolved by Spatial Resolver before any overlay
geometry. Project-resource rows provide coordinates when present but do not
contain geometry columns. Catalogued overlays provide geometry when available;
treat an overlay as contextual evidence, not as a replacement for resolved
resource geometry. Reproject geometries to the plotting CRS
before plotting, and label every rendered layer with its layer name, version,
CRS, and attribution. Report the catalog source and the rows returned.

If the catalog has no matching or available overlay, omit that overlay and
say so explicitly; continue with the authoritative geometry when it is
available. Do not imply that an omitted overlay means that no feature exists.

OpenTerminal plotting of DuckDB query output requires an image with
`geopandas`, `shapely`, and `pyproj`. Keep this requirement limited to the
provided image/runtime: never download runtime tiles or other remote data,
add host paths or mount variables, or provide a model-controlled package
installation path.

## Evidence boundary

Geometry and coordinates are spatial evidence, not narrative labels. Use
Spatial Resolver and catalogued layers for geometry; keep geometry-derived
results separate from authoritative resource metrics and disclose coordinate
or distance assumptions.

## Forbidden shortcuts

- Do not approximate a radius with name matching.
- Do not drop WK scope or units while translating a spatial request to SQL.
- Do not claim that omitted geometry proves an entity is outside the area.
