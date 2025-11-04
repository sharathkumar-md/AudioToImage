# Import the node class from your AudioToConditioning.py file
from .AudioToConditioning import AudioConditioningNode

# --- ComfyUI Registration ---
NODE_CLASS_MAPPINGS = {
    "AudioToConditioning": AudioConditioningNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioToConditioning": "Audio To Conditioning (CLAP)"
}
