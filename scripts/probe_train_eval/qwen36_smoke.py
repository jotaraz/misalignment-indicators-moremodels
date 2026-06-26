"""De-risk Qwen3.6-35B-A3B for probe activation extraction.

Determines: (1) which AutoModel class loads the multimodal MoE, (2) the text
backbone module path, (3) whether the exact `Activations.from_model` forward
call returns 41 hidden_states of [B,S,2048], (4) chat template + <think> handling.
Run on one GPU via srun. Read-only w.r.t. repo state.
"""
import os, sys, traceback
os.environ.setdefault("HF_HOME", "/workspace-vast/kaiwen/hf_cache")
import torch

MODEL = "Qwen/Qwen3.6-35B-A3B"
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | n_gpu {torch.cuda.device_count()}")
import transformers
print("transformers", transformers.__version__)

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
print("tokenizer OK, vocab", tok.vocab_size)

# 1) chat template (mimic a reasoning+text assistant turn)
msgs = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 17 * 23? Think briefly."},
    {"role": "assistant", "content": "<think>17*23 = 17*20 + 17*3 = 340+51 = 391.</think>It's 391."},
]
try:
    ids = tok.apply_chat_template(msgs, add_generation_prompt=False, return_tensors="pt")
    print(f"chat_template OK, seqlen {ids.shape[-1]}")
except Exception as e:
    print("chat_template FAIL:", e)
    ids = tok("hello world", return_tensors="pt").input_ids

# 2) try loaders in order
from transformers import (AutoModelForCausalLM, AutoModel)
loaders = [("AutoModelForCausalLM", AutoModelForCausalLM)]
try:
    from transformers import AutoModelForImageTextToText
    loaders.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
except Exception:
    pass
loaders.append(("AutoModel", AutoModel))

model = None
used = None
for name, cls in loaders:
    try:
        print(f"\n--- trying {name} ---")
        model = cls.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        used = name
        print(f"LOADED via {name}: {type(model).__name__}")
        break
    except Exception as e:
        print(f"{name} failed: {type(e).__name__}: {str(e)[:200]}")

if model is None:
    print("\nNO LOADER WORKED"); sys.exit(1)

# 3) inspect module tree for the text backbone
print("\n=== top-level children ===")
for n, _ in model.named_children():
    print(" ", n)
# find a submodule with .layers
def find_lm(m, prefix=""):
    for n, child in m.named_children():
        if hasattr(child, "layers"):
            return prefix + n, child
        r = find_lm(child, prefix + n + ".")
        if r:
            return r
    return None
lm = find_lm(model)
if lm:
    print(f"text backbone with .layers: '{lm[0]}'  ({len(lm[1].layers)} layers)")

dev = "cuda"
ids = ids.to(dev)
attn = torch.ones_like(ids)

def try_forward(callable_obj, label):
    print(f"\n=== forward via {label} ===")
    for kwargs in [
        dict(output_hidden_states=True, attention_mask=attn, num_logits_to_keep=1, use_cache=False),
        dict(output_hidden_states=True, attention_mask=attn, use_cache=False),
    ]:
        try:
            with torch.no_grad():
                out = callable_obj(ids, **kwargs)
            hs = getattr(out, "hidden_states", None)
            if hs is None:
                print(f"  kwargs={list(kwargs)} -> NO hidden_states attr (type {type(out).__name__})")
                continue
            print(f"  kwargs={list(kwargs)} -> hidden_states: n={len(hs)}, shape={tuple(hs[0].shape)}, dtype={hs[0].dtype}")
            print(f"    layer23 norm={hs[23].float().norm().item():.1f}  layer25 norm={hs[25].float().norm().item():.1f}")
            return True
        except Exception as e:
            print(f"  kwargs={list(kwargs)} -> FAIL {type(e).__name__}: {str(e)[:160]}")
    return False

ok_full = try_forward(model, "full model")
if not ok_full and lm:
    try_forward(lm[1], f"inner backbone ({lm[0]})")

print("\nSUMMARY: loader=", used, "| name_or_path=", getattr(model, "name_or_path", "n/a"))
print("done")
