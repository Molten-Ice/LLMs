from transformers import OlmoeForCausalLM, AutoTokenizer
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load different ckpts via passing e.g. `revision=step10000-tokens41B`
# also check allenai/OLMoE-1B-7B-0924-SFT & allenai/OLMoE-1B-7B-0924-Instruct
model = OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
inputs = tokenizer("Bitcoin is", return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
out = model.generate(**inputs, max_length=64)
print(tokenizer.decode(out[0]))
# Bitcoin is a digital currency that is created and held electronically. No one controls it. Bitcoins aren’t printed, like dollars or euros – they’re produced by people and businesses running computers all around the world, using software that solves mathematical