# Handoff: gpt-oss-120b probe training (babysit to 18/18)

**Owner of this file:** hand-off from a prior agent session. Goal: get the 18-indicator
probe ensemble trained for **gpt-oss-120b**. It keeps crashing on a CUDA OOM; this doc
has everything you need to diagnose, fix, resubmit, and monitor it.

_All paths are on the **mpi-cluster**, not the local Mac._

---

## 1. Goal / definition of done
Train `probe/probes_gptoss_120b/v4_v2_6_combined_v2_span/` to **18/18 indicators**
(each indicator dir contains `span/layer20/` and `span/layer22/` with `detector.pt`).
Layers are **20 / 22** for gpt-oss-120b (36-layer MoE; matches GLM 27/29 depth-wise).

**Current state: 3/18** (`action_concealment`, `adversarial_user_framing`,
`concerns_on_self_existence`). Training is **resumable** — `probe/train.py` skips any
indicator that already has probes, so every resubmit picks up where it left off.

---

## 2. Access
- Cluster login: `ssh jtaraz@login.cluster.is.localnet` (MPI-IS HTCondor cluster).
  - The prior session reused an SSH ControlMaster socket at `/tmp/clcm.sock`; you'll
    want to open your own (`ssh -fN -M -S /tmp/clcm.sock jtaraz@login.cluster.is.localnet`)
    or just connect normally. **Do not run compute on the login node.**
- Repo: `/fast/jtaraz/LIARS/misalignment-indicators-moremodels`
- Training python env: `/fast/jtaraz/envs/probe/bin/python`
- HF weights cache: `HF_HOME=/fast/jtaraz/huggingface` (gpt-oss-120b already downloaded).
- Scheduler: HTCondor. Submit with `condor_submit_bid <bid> ...`; watch with
  `condor_q jtaraz`; post-mortem with `condor_history <jobid> -af JobStatus ExitCode`.

---

## 3. The problem
gpt-oss training **crashes ~immediately into the 4th indicator (stuck at 3/18)** with:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.92 GiB.
GPU 0 has 79.18 GiB total, ~6.7 GiB free, ~71.3 GiB allocated by PyTorch,
~0.4 GiB reserved-but-unallocated.
```

Crashed jobs (all exit-code 1, all at 3/18): `17379227`, `17379467`, `17379584`.

**Why:** gpt-oss-120b's weights fill GPU 0 to ~71/79 GB across the 4×H100 shard, leaving
too little headroom for the **eager-attention** O(seq²) softmax spike on a long training
sample. The gpt-oss loader **hardcodes `attn_implementation="eager"`** (see
`ood_misalignment_eval/deception-detection/deception_detection/models.py`,
`get_gpt_oss_model_and_tokenizer`) because gpt-oss uses harmony/custom attention — so the
**`sdpa` trick that fixed DeepSeek-R1 is NOT available here.**

Note the fragmentation is already handled (only ~0.4 GiB reserved-unallocated now); this
is a **genuine VRAM-capacity** shortfall, not fragmentation.

---

## 4. What's already been tried (so you don't repeat it)
1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — **added** to
   `scripts/cluster/run_train.sh` (the older var name; the newer `PYTORCH_ALLOC_CONF`
   was already there but this torch build reads the CUDA-prefixed one). This eliminated
   the *fragmentation* OOMs (it got gpt-oss from 1/18 → 3/18) but not the capacity OOM.
2. `request_memory = 262144` (256 GB host RAM) — fine, not the bottleneck (GPU VRAM is).
3. **Per-sample OOM-skip** added to `probe/train.py` `extract_activations()` — wraps the
   `Activations.from_model(...)` call in `try/except torch.cuda.OutOfMemoryError` to skip
   the offending sample. **This did NOT fire** (0 skips logged before the crash). Two
   likely reasons — **please verify**:
   - **Exception-class mismatch:** the raised class is `torch.OutOfMemoryError`; the
     `except` catches `torch.cuda.OutOfMemoryError`. In recent torch these are the same
     object, but confirm in this env (`/fast/jtaraz/envs/probe/bin/python -c "import
     torch; print(torch.OutOfMemoryError is torch.cuda.OutOfMemoryError)"`). If not the
     same, broaden to `except (torch.OutOfMemoryError, RuntimeError)` and re-check the msg.
   - **Wrong loop / uncatchable point:** the `.err` shows a `tqdm` `"Extracting
     activations: .../1885"` — confirm that bar comes from the *same* per-sample loop you
     wrapped and not a batched path inside `Activations.from_model`. A hard CUDA OOM can
     also corrupt the context so `empty_cache()` in the `except` can't recover — in which
     case skipping won't help and you need more VRAM.

`.bak` backups of the edited files exist on the cluster (`*.py.bak`).

---

## 5. Recommended fix (in priority order)
1. **Bump GPUs to 6 (or 8) × H100.** This is the most reliable fix — fewer weight shards
   per GPU → more free VRAM for the attention spike. gpt-oss-120b bf16 ≈ 240 GB; on 6
   GPUs that's ~40 GB/GPU of weights, leaving ~40 GB headroom (vs ~7 GB on 4). Resubmit:
   ```bash
   cd /fast/jtaraz/LIARS/misalignment-indicators-moremodels
   condor_submit_bid 100 scripts/cluster/train.sub \
     -append 'arguments = "'"$PWD"'/scripts/cluster/run_train.sh gptoss-120b 20 22"' \
     -append 'request_gpus = 6' \
     -append 'request_memory = 262144' \
     -append 'output = '"$PWD"'/logs/train_gptoss_120b.out' \
     -append 'error  = '"$PWD"'/logs/train_gptoss_120b.err' \
     -append 'log    = '"$PWD"'/logs/train_gptoss_120b.log'
   ```
   (It resumes from 3/18. If 6×H100 won't schedule, try 8, or lower the condor `bid`.)
2. **Also fix the OOM-skip** (belt & suspenders) — broaden the `except` in
   `probe/train.py extract_activations()` to `(torch.OutOfMemoryError, RuntimeError)` and
   `torch.cuda.empty_cache()` + `gc.collect()` so a single long sample can't kill the run
   even with more GPUs.
3. If VRAM is still tight, consider **capping the per-sample sequence length** for gpt-oss
   extraction (truncate the longest dialogues) — code change in the extraction loop.

---

## 6. How to monitor
```bash
cd /fast/jtaraz/LIARS/misalignment-indicators-moremodels
condor_q jtaraz -nobatch | grep gptoss                 # running? (ST: R=run, I=idle, H=held)
ls probe/probes_gptoss_120b/v4_v2_6_combined_v2_span/   # indicator count (target 18)
tail -5 logs/train_gptoss_120b.out                      # progress ([diag] sample ...)
grep -iE "out of memory|Skipping sample.*OOM" logs/train_gptoss_120b.err | tail
condor_history <jobid> -af JobStatus ExitCode           # 4/1 = crashed; 4/0 = done
```
Steady state = a new indicator dir appearing every ~15–40 min. If `condor_q` shows it
gone and count < 18 → it crashed; check `.err`, apply a fix from §5, resubmit.

---

## 7. Context: the rest of the effort (for situational awareness — NOT your job)
- **Probe training for other models is healthy:** DeepSeek-R1 (`sdpa`) and gemma-2-27b
  (`expandable_segments` + 4×H100, eager) were both climbing cleanly toward 18/18 when
  this was written. Only gpt-oss is stuck.
- **gpt-oss neutral activations: DONE** (`probe/data/neutral/gptoss/activations/layer{20,21,22}.pt`,
  100k tokens/layer). Do not redo.
- **gpt-oss dev-eval behavioral smoke: DONE but NOT GT-judged** — 5 behaviors ×
  {misaligned,benign} exist under `bloom/bloom-results/*gpt_oss_120b/` (gpt-oss engages
  tools natively via harmony). If asked, GT-judge them with
  `cd bloom && PYBIN=/fast/jtaraz/envs/bloom/bin/python bash cross_model_gt.sh gpt_oss_120b`.
- Uncommitted cluster edits from the prior session exist in `probe/train.py`,
  `models.py`, `probe/neutral/extract_activations.py`, and `scripts/cluster/*` (with
  `.bak` files). They are working code; don't revert without reason.

---

_Bottom line: gpt-oss is 3/18, crashing on an eager-attention VRAM OOM that `sdpa` can't
fix and the OOM-skip didn't catch. Give it more GPUs (6–8×H100), fix the skip's exception
class, resubmit, and babysit it to 18/18._
