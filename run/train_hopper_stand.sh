export PYTHONPATH="${PYTHONPATH}:`pwd`"

# python3 scripts/generate_dmc_video.py --env hopper_stand --episode_len 60
python3 train_gaifo.py task=hopper_stand
