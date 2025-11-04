import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor
import librosa
import os

# --- 1. UPDATED: Transformer-Based Projection Network ---
class ProjectionNetwork(nn.Module):
    """
    A Transformer-based network to project a single audio embedding
    into a sequence of 77 text-conditioning embeddings.
    """
    def __init__(self, audio_embedding_dim=512, text_embedding_dim=768, num_decoder_layers=4, nhead=4):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim

        # Input projection to match the text embedding dimension
        self.input_proj = nn.Linear(audio_embedding_dim, text_embedding_dim)

        # A learnable "start of sequence" token
        self.start_token = nn.Parameter(torch.randn(1, 1, text_embedding_dim))

        # Positional encodings for the sequence of 77 tokens
        self.positional_encoding = nn.Parameter(torch.randn(1, 77, text_embedding_dim))

        # The Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=text_embedding_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

    def forward(self, audio_embedding):
        # 1. Project the audio embedding to match the text dimension
        memory = self.input_proj(audio_embedding).unsqueeze(1) # Shape: [batch_size, 1, 768]

        # 2. Create the target sequence for the decoder
        batch_size = audio_embedding.shape[0]
        tgt_seq = self.start_token.expand(batch_size, 77, -1)

        # 3. Add positional information
        tgt_with_pos = tgt_seq + self.positional_encoding

        # 4. Generate the full sequence using the Transformer Decoder
        output_sequence = self.transformer_decoder(tgt_with_pos, memory)

        return output_sequence

# --- 2. The Main Custom Node Class ---
class AudioConditioningNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_file_path": ("STRING", {"default": "C:/path/to/your/audio.wav"}),
                "projection_network_path": ("STRING", {"default": "C:/path/to/your/projection_network.pth"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "generate_conditioning"
    CATEGORY = "MyAudioProject" # This will be the category in ComfyUI's right-click menu

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clap_model = None
        self.clap_processor = None
        self.projection_network = None
        self.loaded_proj_path = "" # Keep track of the loaded model path

    def load_models(self, model_path):
        # Load CLAP model (only once)
        if self.clap_model is None:
            print("Loading CLAP model...")
            self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
            self.clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            self.clap_model.eval()
            print("CLAP model loaded.")

        # Load Projection Network, but only if it's not loaded or the path has changed
        if self.projection_network is None or self.loaded_proj_path != model_path:
            print(f"Loading Projection Network from: {model_path}")
            
            clean_model_path = model_path.strip().strip('"')
            if not os.path.exists(clean_model_path):
                raise FileNotFoundError(f"Projection network not found at: {clean_model_path}")

            self.projection_network = ProjectionNetwork().to(self.device)
            self.projection_network.load_state_dict(torch.load(clean_model_path, map_location=self.device))
            self.projection_network.eval()
            self.loaded_proj_path = model_path # Update the stored path
            print("Projection Network loaded.")

    def generate_conditioning(self, audio_file_path, projection_network_path):
        self.load_models(projection_network_path)

        clean_path = audio_file_path.strip().strip('"')
        
        print(f"Processing audio file: {clean_path}")
        waveform, sr = librosa.load(clean_path, sr=48000, mono=True)
        # Note: 'audios' expects a list of waveforms
        inputs = self.clap_processor(text=None, audios=[waveform], return_tensors="pt", sampling_rate=48000)
        audio_features = inputs['input_features'].to(self.device)
        
        with torch.no_grad():
            clap_embedding = self.clap_model.get_audio_features(input_features=audio_features)
            # The new network directly outputs the correct [1, 77, 768] sequence
            projected_sequence = self.projection_network(clap_embedding)
        
        print("Formatting conditioning output for ComfyUI.")
        # --- UPDATED: Correct conditioning format for last_hidden_state ---
        # The second element in the list is an empty dictionary.
        final_conditioning = [[projected_sequence, {}]]
        
        return (final_conditioning,)
