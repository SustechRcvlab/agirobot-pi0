uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=checkpoints/pi0_fast_libero/my_experiment/20000
uv run scripts/train.py policy:checkpoint --policy.config=pi0_fast_libero --policy.dir=checkpoints/
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_base_agibot_lora --exp-name=agibot --overwrite
