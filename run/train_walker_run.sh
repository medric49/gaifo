export PYTHONPATH="${PYTHONPATH}:`pwd`"

# python3 scripts/generate_dmc_video.py --env walker_run --episode_len 60
python3 train_gaifo.py task=walker_run
