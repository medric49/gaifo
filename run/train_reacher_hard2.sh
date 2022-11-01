export PYTHONPATH="${PYTHONPATH}:`pwd`"

python3 scripts/generate_dmc_video.py --env reacher_hard2 --episode_len 60
python3 train_gaifo.py task=reacher_hard2


