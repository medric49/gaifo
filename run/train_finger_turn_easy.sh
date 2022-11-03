export PYTHONPATH="${PYTHONPATH}:`pwd`"

# python3 scripts/generate_dmc_video.py --env finger_turn_easy --episode_len 60
python3 train_gaifo.py task=finger_turn_easy
