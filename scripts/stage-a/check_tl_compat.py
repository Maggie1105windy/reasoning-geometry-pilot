import torch
import transformers
import transformer_lens
from importlib.metadata import version
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_official_model_name

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("transformer_lens", version("transformer-lens"))
print("official_name", get_official_model_name("pythia-2.8b"))

# Constructor/API inspection only: do not download or instantiate weights here.
assert get_official_model_name("pythia-2.8b") == "EleutherAI/pythia-2.8b"
assert hasattr(HookedTransformer, "from_pretrained")
print("compatibility_import_check=PASS")
