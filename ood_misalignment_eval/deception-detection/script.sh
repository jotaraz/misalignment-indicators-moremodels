cd /workspace-vast/kaiwen/persona_vectors/data_generation/deception-detection

.venv/bin/python -m deception_detection.scripts.generate_rollouts --dataset_partial_id=roleplaying__plain --model_name=glm-9b-flash --use_api=False --grade_rollouts=True --grader_api=anthropic --add_to_repo=True --num=1 --max_prompts=200

.venv/bin/python -m deception_detection.scripts.generate_rollouts --dataset_partial_id=insider_trading__upscale --model_name=glm-9b-flash --use_api=False --grade_rollouts=True --grader_api=anthropic --add_to_repo=True --num=1 --max_prompts=200

.venv/bin/python -m deception_detection.scripts.generate_rollouts --dataset_partial_id=insider_trading_doubledown__upscale --model_name=glm-9b-flash --use_api=False --grade_rollouts=True --grader_api=anthropic --add_to_repo=True --num=1 --max_prompts=200

.venv/bin/python -m deception_detection.scripts.generate_rollouts --dataset_partial_id=sandbagging_v2__wmdp_mmlu --model_name=glm-9b-flash --use_api=False --grade_rollouts=True --grader_api=anthropic --add_to_repo=True --num=1 --max_prompts=200

.venv/bin/python -m deception_detection.scripts.generate_rollouts --dataset_partial_id=alpaca__plain --model_name=glm-9b-flash --use_api=False --grade_rollouts=False --add_to_repo=True --num=1 --max_prompts=1000
