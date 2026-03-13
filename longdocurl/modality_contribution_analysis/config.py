"""
Configuration and constants for modality contribution analysis - LongDocURL benchmark.
"""

# Models to evaluate - using vision-capable models for image support
MODELS_TO_EVALUATE = [
    "gpt-4o-mini",  # Vision-capable model
    "qwen3-vl:8b",
    "qwen3-vl:30b",
    "gemma3:4b",
    "gemma3:27b",
    "llava:13b",
    "minicpm-v:8b"
]

# Modalities that have images (these will be sent as actual images to the model)
IMAGE_MODALITIES = {'image', 'table'}

# Map LongDocURL evidence_sources to internal modality types
# LongDocURL uses: Text, Table, Figure, Layout
MODALITY_MAPPING = {
    'Figure': ['image'],
    'Table': ['table'],
    'Text': ['text', 'list'],
    'Layout': ['header', 'footer', 'page_number', 'page_footnote', 'aside_text'],
}

# Modalities that should be treated as a single logical group
# These will NOT be expanded into individual modalities for subset generation
# But will be expanded when filtering edges (to match any of the internal modalities)
GROUPED_MODALITIES = {
    'layout': ['header', 'footer', 'page_number', 'page_footnote', 'aside_text'],  # Layout
    'plain_text': ['text', 'list'],  # Text
}
