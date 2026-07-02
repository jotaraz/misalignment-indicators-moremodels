# Handoff: Llama-3.3-70B-Instruct (vanilla) probe training

**Goal:** train the 18-indicator probe ensemble for the **vanilla Llama-3.3-70B-Instruct**
(registry key `llama-70b-3.3` → `probe/probes_llama_70b_3_3/`, layers **48 / 50**).
This is the **non-reasoning meta-llama** model — *not* DeepSeek-R1-Distill (`llama-70b-r1`),
which is a separate, already-working job.

_All paths on the **mpi-cluster** (`/fast/jtaraz/LIARS/misalignment-indicators-moremodels`)._

**Current state: 0/18.** `probe/probes_llama_70b_3_3/` does not exist yet.

---

## What happened (why the old job was killed)
Job `17370467` ran for **5.5 days doing nothing** and was **killed** on 2026-07-02.
It hung immediately at model download:
```
Loading llama-70b-3.3 model and tokenizer...
Fetching 30 files:   0%|          | 0/30      <-- frozen here for 5.5 days
```
**Root cause:** the weights id `meta-llama/Llama-3.3-70B-Instruct` is **gated
(`gated: manual`)** and the HF token does **not** have approved download access —
only ~96 KB of metadata ever landed in the cache, zero weights. The HF hub download
blocks at the license gate and hangs forever, holding a 2-GPU reservation. It never
loaded the model or trained a single indicator.

(Contrast: DeepSeek-R1 training works fine because `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`
is **ungated**.)

---

## The fix (recommended — one line)
Point the weights at the **ungated `unsloth/Llama-3.3-70B-Instruct` mirror** (identical
weights, no license gate). Notably the **tokenizer is already loaded from unsloth**, so
this just makes the weights consistent:

`ood_misalignment_eval/deception-detection/deception_detection/models.py`, line ~280:
```python
# before (gated, hangs):
ModelName.LLAMA_70B_3_3: "meta-llama/Llama-3.3-70B-Instruct",
# after (ungated mirror):
ModelName.LLAMA_70B_3_3: "unsloth/Llama-3.3-70B-Instruct",
```
(line ~266 already has `tokenizer_name = "unsloth/Llama-3.3-70B-Instruct"`.)

**Alternative:** request access to `meta-llama/Llama-3.3-70B-Instruct` on HuggingFace and
wait for manual approval, then no code change is needed. The unsloth mirror is faster.

Sanity-check the mirror downloads before a full run:
```bash
set -a; source bloom/.env; set +a
/fast/jtaraz/envs/probe/bin/python -c "from huggingface_hub import snapshot_download; \
print(snapshot_download('unsloth/Llama-3.3-70B-Instruct', allow_patterns=['config.json']))"
```

---

## Resubmit (after the weights id is fixed)
Use the **same config as the working DeepSeek-R1 job** (same 70B arch, layers 48/50):
4×H100, `sdpa` attention, 256 GB RAM. `run_train.sh` already sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the llama loader honors `ATTN_IMPL`,
and `train.py` has a per-sample OOM-skip.
```bash
cd /fast/jtaraz/LIARS/misalignment-indicators-moremodels
condor_submit_bid 100 scripts/cluster/train.sub \
  -append 'arguments = "'"$PWD"'/scripts/cluster/run_train.sh llama-70b-3.3 48 50"' \
  -append 'request_gpus = 4' \
  -append 'request_memory = 262144' \
  -append 'environment = "ATTN_IMPL=sdpa"' \
  -append 'output = '"$PWD"'/logs/train_llama_70b_3_3.out' \
  -append 'error  = '"$PWD"'/logs/train_llama_70b_3_3.err' \
  -append 'log    = '"$PWD"'/logs/train_llama_70b_3_3.log'
```
Training is resumable (`train.py` skips indicators that already have probes), so it's safe
to resubmit after any interruption. Expect ~30–40 min/indicator on the 70B → several hours.

---

## Monitoring
```bash
condor_q jtaraz -nobatch | grep llama_70b_3_3          # ST: R=running, I=idle, H=held
ls probe/probes_llama_70b_3_3/v4_v2_6_combined_v2_span/ # indicator count (target 18)
tail -5 logs/train_llama_70b_3_3.out                   # [diag] sample ... = extracting
grep -iE "out of memory|Fetching 30 files|Skipping sample" logs/train_llama_70b_3_3.err | tail
```
**Watch the first ~2 min:** if `.err` shows `Fetching NN files: 0%` and stays frozen, the
download is stuck again (gate/mirror problem) — kill and fix the id, don't let it hang.

---

## Don't redo / context
- **Rollouts + GT for llama-3.3-70b are DONE** — they go through **OpenRouter**
  (`openrouter/meta-llama/llama-3.3-70b-instruct`, no HF weights needed), including the
  `candid_tools` scratchpad smoke where it drove full simenv tool loops (self-pres 10 /
  code-sabotage 8 tool rounds). Dirs: `bloom/bloom-results/*_llama_3_3_70b`.
- **Neutral activations: NOT done** for llama-3.3-70b (needs the HF weights on GPU, same
  as training). Do them *after* training works, via `scripts/cluster/run_neutral_extract.sh`
  (add a `llama33` entry to `MODEL_CONFIGS` in `probe/neutral/extract_activations.py`,
  mirroring the deepseek entry: `ModelName.LLAMA_70B_3_3`, path = unsloth mirror,
  layers [48,49,50]).
- Access: `ssh jtaraz@login.cluster.is.localnet`; python env `/fast/jtaraz/envs/probe/bin/python`;
  `HF_HOME=/fast/jtaraz/huggingface`; HF token in `bloom/.env`. Don't compute on the login node.
- Uncommitted cluster edits from prior sessions exist (`.bak` backups present); they're
  working code (sdpa support, OOM-skip, expandable_segments) — don't revert.

_Bottom line: 0/18, old job killed (hung on gated meta-llama download). Switch the weights
id to `unsloth/Llama-3.3-70B-Instruct`, resubmit with the DeepSeek-R1 config
(4×H100 + sdpa + 256 GB), and babysit to 18/18._
