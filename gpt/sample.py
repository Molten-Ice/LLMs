import os
import time
import pickle
import torch
import tiktoken
from model import GPTConfig, GPT

assert torch.cuda.is_available(), 'CUDA is not available'

temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 3
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster

init_from = 'gpt2-xl' # 'gpt2-xl'
start = "What is the answer to life, the universe, and everything?" # "\n" # or "<|endoftext|>" or etc.
# start = 'Hello world!'
# start = "Asimov foundation trilogy:"
max_new_tokens = 100

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    print('Compiling model...')
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

max_new_tokens = 100
print(f'x.shape: {x.shape}, max_new_tokens: {max_new_tokens}')

# My kv caching actually is slower than the original :) (lol)

# For 100 tokens:
# Original: 0.8398 seconds
# My kv caching: 1.6832 seconds

# Successful slowdown :[]

num_sequences = 5

output = []
with torch.no_grad():
    with torch.amp.autocast(device_type=device, dtype=ptdtype):
        for i in range(num_sequences):
            print(f'------ Sequence {i+1} ------')
            t0 = time.time()
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(f'Time taken: {time.time() - t0:.4f} seconds')
            output.append(f'Sequence {i+1}: {decode(y[0].tolist())}')

print('\n--------------\n'.join(output))

# --------------
# Sequence 4: What is the answer to life, the universe, and everything?

# I think the answer is pretty simple. And I think this answer is the most important thing that we need to have.

# If that is correct then we need to build the tools to discover the answer. That means building intelligent machines that are intelligent in themselves. In other words, building a machine that is intelligence. As intelligent as our computers are, there's no intelligence that we know about that can understand itself completely and also solve problems that we can understand in an understandable way.
