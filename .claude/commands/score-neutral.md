Score neutral dialogues with probes and tuned thresholds via SLURM.

Arguments: $ARGUMENTS
- First argument: probe version (e.g. v2_4_span, v3_v2_5_span). Required.
- Second argument: thresholds file path relative to probe_eval/results/. If not provided, defaults to probe_eval/results/<probe_version>/tuned_thresholds.json.
- Optional flags: --max-dialogues N (default 3000), --layer N (default 27), --thresholds-version VERSION (default fpr_0.02)

Examples:
  /score-neutral v2_4_span
  /score-neutral v2_4_span probe_eval/results/v2_4_span/tuned_thresholds_test.json
  /score-neutral v3_v2_5_span --max-dialogues 500 --layer 28

Instructions:

1. Parse the arguments from $ARGUMENTS. Extract:
   - probe_version (first positional arg, REQUIRED - error if missing)
   - thresholds_file (second positional arg if it ends in .json, else default)
   - --max-dialogues (default: 3000)
   - --layer (default: 27)
   - --thresholds-version (default: fpr_0.02)

2. Set paths:
   - BASE_DIR = /workspace-vast/kaiwen/persona_vectors/data_generation
   - PROBES_DIR = ${BASE_DIR}/probe/probes/${probe_version}
   - DIALOGUES_PATH = ${BASE_DIR}/probe/data/neutral/dialogues_filtered_v3.json
   - THRESHOLDS_FILE = if second arg provided, ${BASE_DIR}/${thresholds_file}; else ${BASE_DIR}/probe_eval/results/${probe_version}/tuned_thresholds.json
   - OUTPUT_PATH = ${BASE_DIR}/probe/data/neutral/false_positives_${probe_version}_tuned.json

3. Verify the probes directory and thresholds file exist. If not, report the error and stop.

4. Create a SLURM script under ${BASE_DIR}/logs/score_neutral_${probe_version}_<timestamp>/ with:
   - Job name: score_neutral_${probe_version}
   - GPU: 1, partition: general, mem: 128G
   - Activate venv: ${BASE_DIR}/ood_misalignment_eval/deception-detection/.venv/bin/activate
   - Set HF_HOME=/workspace-vast/pretrained_ckpts and HF_HUB_OFFLINE=1
   - Run: python -m probe.neutral.score_neutral with all the args above

5. Submit with sbatch and report the job ID, output path, and monitoring command.
