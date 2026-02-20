"""
Configuration and constants for modality contribution analysis (WITH IMAGES).
"""

# Models to evaluate - using vision-capable models for image support
MODELS_TO_EVALUATE = [
    #  "gpt-4o-mini",  # Vision-capable model (upgraded from gpt-4o-mini)
    #  "qwen3-vl:8b",
    #  "qwen3-vl:30b",
     "gemma3:4b",
    #  "gemma3:27b",
    #  "llava:13b",
    #  "minicpm-v:8b"
]

# Modalities that have images (these will be sent as actual images to the model)
IMAGE_MODALITIES = {'image', 'table'}

# Map evidence_sources to modality types (benchmark modality -> internal modalities to filter)
MODALITY_MAPPING = {
    'Chart': ['image'],
    'Figure': ['image'],
    'Table': ['table'],
    'Pure-text (Plain-text)': ['text', 'list'],
    'Generalized-text (Layout)': ['header', 'footer', 'page_number', 'page_footnote'],
}

# Modalities that should be treated as a single logical group (Option B)
# These will NOT be expanded into individual modalities for subset generation
# But will be expanded when filtering edges (to match any of the internal modalities)
GROUPED_MODALITIES = {
    'layout': ['header', 'footer', 'page_number', 'page_footnote'],  # Generalized-text (Layout)
    'plain_text': ['text', 'list'],  # Pure-text (Plain-text)
}
