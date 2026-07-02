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
It hung immediately at model download and never recovered:
```
Loading llama-70b-3.3 model and tokenizer...
Fetching 30 files:   0%|          | 0/30      <-- frozen here for 5.5 days
```

**Root cause — a STALLED weight download, NOT an auth problem.** (Corrected diagnosis.)
The HF token in `bloom/.env` (`HF_TOKEN`) **does have approved access** to the gated
`meta-llama/Llama-3.3-70B-Instruct` — verified: `hf_hub_download(repo, "config.json")`
with that token succeeds. The job hung because the inline multi-file weight fetch
(~30 safetensors, ~140 GB, to `HF_HOME=/fast/jtaraz/huggingface`) stalled at 0% and the
training process never retried — the **same class of stall the gpt-oss download hit**,
which only resolved with a dedicated fetch. So: don't chase auth; just get the weights
onto disk with a resumable download, then train.

---

## The fix
**1. Pre-download the weights with a dedicated CPU job (resumable, token works).**
Already launched on 2026-07-02 as job **`17379746`**:
```bash
condor_submit_bid 100 scripts/cluster/download.sub \
  -append 'arguments = "'"$PWD"'/scripts/cluster/run_hf_download.sh meta-llama/Llama-3.3-70B-Instruct"' \
  -append 'request_gpus = 0' \
  -append 'output = '"$PWD"'/logs/dl_llama33.out' -append 'error = '"$PWD"'/logs/dl_llama33.err' \
  -append 'log = '"$PWD"'/logs/dl_llama33.log'
```
Watch it finish (it prints `DOWNLOAD COMPLETE`); confirm the cache is full:
```bash
tail -3 logs/dl_llama33.out
du -sh /fast/jtaraz/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct   # want ~140 GB
```
If it stalls again (size stuck, no progress for a while), kill and resubmit the download —
it resumes from partial. **Fallback only if it repeatedly won't complete:** switch the
weights id to the ungated mirror `unsloth/Llama-3.3-70B-Instruct` in
`ood_misalignment_eval/deception-detection/deception_detection/models.py` (line ~280,
`ModelName.LLAMA_70B_3_3`); the tokenizer (line ~266) already uses that mirror. **No code
change is needed for the normal path** — the meta-llama weights work with the token.

**2. Once the weights are cached, resubmit training** — same config as the working
DeepSeek-R1 job (same 70B arch, layers 48/50): 4×H100, `sdpa`, 256 GB RAM. `run_train.sh`
already sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, the llama loader honors
`ATTN_IMPL`, and `train.py` has a per-sample OOM-skip.
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
Training is resumable (`train.py` skips indicators that already have probes). Expect
~30–40 min/indicator on the 70B → several hours to 18/18. **Weights load from the /fast
cache instantly** once the pre-download is done, so `Fetching …` should NOT reappear.

---

## Monitoring
```bash
condor_q jtaraz -nobatch | grep -E "dl_llama33|llama_70b_3_3"   # download / training
ls probe/probes_llama_70b_3_3/v4_v2_6_combined_v2_span/         # indicator count (target 18)
tail -5 logs/train_llama_70b_3_3.out                            # [diag] sample ... = extracting
grep -iE "out of memory|Fetching .* files|Skipping sample" logs/train_llama_70b_3_3.err | tail
```
**Watch the first ~2 min of training:** if `.err` shows `Fetching NN files: 0%` frozen,
the pre-download didn't fully land — fix the cache before letting it hang again.

---

## Don't redo / context
- **Rollouts + GT for llama-3.3-70b are DONE** — via **OpenRouter**
  (`openrouter/meta-llama/llama-3.3-70b-instruct`, no HF weights needed), including the
  `candid_tools` scratchpad smoke where it drove full simenv tool loops (self-pres 10 /
  code-sabotage 8 tool rounds). Dirs: `bloom/bloom-results/*_llama_3_3_70b`.
- **Neutral activations: NOT done** for llama-3.3-70b (needs the HF weights on GPU, same
  as training). Do them *after* training works: add a `llama33` entry to `MODEL_CONFIGS`
  in `probe/neutral/extract_activations.py` (mirror the `deepseek` entry:
  `ModelName.LLAMA_70B_3_3`, path `meta-llama/Llama-3.3-70B-Instruct`, layers [48,49,50]),
  then `run_neutral_extract.sh llama33`.
- Access: `ssh jtaraz@login.cluster.is.localnet`; python env `/fast/jtaraz/envs/probe/bin/python`;
  `HF_HOME=/fast/jtaraz/huggingface`; `HF_TOKEN` in `bloom/.env` (has meta-llama access).
  Don't compute on the login node.
- Uncommitted cluster edits from prior sessions exist (`.bak` backups present); working
  code (sdpa support, OOM-skip, expandable_segments) — don't revert.

_Bottom line: 0/18, old job killed (hung on a stalled weight download — token access is
fine). Weights are being pre-downloaded (job 17379746); once cached, resubmit training with
the DeepSeek-R1 config (4×H100 + sdpa + 256 GB) and babysit to 18/18._
