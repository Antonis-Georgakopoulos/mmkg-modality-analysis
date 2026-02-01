"""
Configuration and constants for modality contribution analysis.
"""

# # Models to evaluate
# MODELS_TO_EVALUATE = [
#     "gpt-4o-mini",
#     "gemma3:27b",
#     "qwen3:30b",
#     "mistral:7b",
#     "deepseek-r1:14b",
#     "llama3.2:3b",
#     "phi4"
# ]


# # Models to evaluate
MODELS_TO_EVALUATE = [
     "gpt-4o-mini",
     "gemma3"
]


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
