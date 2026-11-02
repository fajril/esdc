-- Canonical reference seed version 20261028. Updates preserve IDs and local extra codes.
INSERT INTO r_institution (id,institution,description,created_by,updated_by) VALUES
(1,'MESDM','POD I atau POD Pertama','FA:reference-seed-2026-10-27','FA:reference-seed-2026-10-27'),
(2,'BPPKA','POD prior to 2003','FA:reference-seed-2026-10-27','FA:reference-seed-2026-10-27'),
(3,'BP Migas','POD 2003 to 2012','FA:reference-seed-2026-10-27','FA:reference-seed-2026-10-27'),
(4,'SKK Migas','POD 2013 onward','FA:reference-seed-2026-10-27','FA:reference-seed-2026-10-27'),
(5,'BPMA','POD in Aceh Province','FA:reference-seed-2026-10-27','FA:reference-seed-2026-10-27')
ON CONFLICT(id) DO UPDATE SET institution=excluded.institution,description=excluded.description,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now'),updated_by=excluded.updated_by WHERE r_institution.institution IS NOT excluded.institution OR r_institution.description IS NOT excluded.description;
INSERT INTO r_pod_type (id,pod_type,description,created_by,updated_by) VALUES
(1,'POD I','the first POD','FA:reference-seed-2026-10-27','FA:reference-seed-2026-10-27'),
(2,'POD/Waterflood/EOR','Approved by other than MESDM','FA:reference-seed-2026-10-27','FA:reference-seed-2026-10-27'),
(3,'POFD/OPL/OPLL','Further Development. Might be POFD, OPL, OPLL','FA:reference-seed-2026-10-27','FA:reference-seed-2026-10-27'),
(4,'POP','Put on Production','FA:reference-seed-2026-10-27','FA:reference-seed-2026-10-27')
ON CONFLICT(id) DO UPDATE SET pod_type=excluded.pod_type,description=excluded.description,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now'),updated_by=excluded.updated_by WHERE r_pod_type.pod_type IS NOT excluded.pod_type OR r_pod_type.description IS NOT excluded.description;
INSERT INTO r_pod_stage (id,stage_name,stage_definition,created_by,updated_by) VALUES
(1,'Stage I','Tahap setelah persetujuan POD berupa pelaksanaan FEED atau persiapan Engineering, Procurement, Construction and Installation (EPCI) dan/atau pengeboran dan lapangan belum onstream.','FA:monitoring-stage','FA:monitoring-stage'),
(2,'Stage II','Tahap pelaksanan EPCI dan lapangan yang belum onstream.','FA:monitoring-stage','FA:monitoring-stage'),
(3,'Stage III','Tahap pada saat lapangan telah onstream dan masih terdapat program kerja POD yang belum selesai.','FA:monitoring-stage','FA:monitoring-stage'),
(4,'Stage IV','Tahap pada saat lapangan telah onstream dan pelaksanaan program kerja POD telah selesai seluruhnya sampai dengan berakhirnya masa berlaku KKS atau economic limit suatu pengembangan lapangan.','FA:monitoring-stage','FA:monitoring-stage')
ON CONFLICT(id) DO UPDATE SET stage_name=excluded.stage_name,stage_definition=excluded.stage_definition,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now'),updated_by=excluded.updated_by WHERE r_pod_stage.stage_name IS NOT excluded.stage_name OR r_pod_stage.stage_definition IS NOT excluded.stage_definition;
