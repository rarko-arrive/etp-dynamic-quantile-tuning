"""Shared segmentation labels used by the panel, experiment framework, and apps.

`mode` (Contract/Spot from ``load_class_bucket``) stays the hybrid v0 dimension.
``mode_tier`` splits Contract into C-1/C-2/C-3 for taxonomy experiments without
changing the production hybrid default.
"""

# load_class → display tier. C-2 = Secondary (backup); C-3 = Rate On File.
MODE_TIER_MAP = {
    "Contract - Primary": "C-1 · Primary",
    "Contract - Secondary": "C-2 · Backup",
    "Contract - Rate On File/Backup": "C-3 · Rate On File",
    "Spot": "Spot",
}
MODE_TIER_ORDER = tuple(MODE_TIER_MAP.values())
