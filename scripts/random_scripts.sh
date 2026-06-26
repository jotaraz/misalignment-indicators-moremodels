source /workspace-vast/kaiwen/envs/persona_vector/bin/activate
cd /workspace-vast/kaiwen/persona_vectors/data_generation/llm_eval_data/deception-detection
source .venv/bin/activate
uv pip install 
srun -p dev,overflow --qos=dev --cpus-per-task=8 --gres=gpu:1 --job-name=D_kaiwen --mem=48G --pty bash
squeue --me
bloom run bloom-data