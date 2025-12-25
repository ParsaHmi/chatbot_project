from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import warnings
warnings.filterwarnings("ignore")

# torch.backends.cudnn.enabled = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu" # "cpu" or "cuda"

torch_dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else (
    torch.float16 if device.startswith("cuda") else torch.float32
)

print("Device:", device)
print("Torch DType:", torch_dtype)