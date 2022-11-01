import argparse
import warnings
import sys

import context_changers
import dmc
import utils
from drqv2 import DrQV2Agent
from pathlib import Path

warnings.filterwarnings('ignore', category=DeprecationWarning)

env_data = {
    'reacher_hard2': ('reacher_hard', 'exp_local/reacher_hard/1/snapshot.pt', 'domain_xmls/reacher.xml', context_changers.ReacherHardWCContextChanger),
    'reacher_hard': ('reacher_hard', 'exp_local/reacher_hard/1/snapshot.pt', 'domain_xmls/reacher.xml', context_changers.ReacherHardContextChanger),
    'finger_turn_easy': ('finger_turn_easy', 'exp_local/finger_turn_easy/1/snapshot.pt', None, context_changers.NullContextChanger),
    'hopper_stand': ('hopper_stand', 'exp_local/hopper_stand/1/snapshot.pt', None, context_changers.NullContextChanger),
    'cartpole_swingup': ('cartpole_swingup', 'exp_local/cartpole_swingup/1/snapshot.pt', None, context_changers.NullContextChanger),
    'reacher_easy': ('reacher_easy', 'exp_local/reacher_easy/1/snapshot.pt', None, context_changers.NullContextChanger),
    'walker_run': ('walker_run', 'exp_local/walker_run/1/snapshot.pt', None, context_changers.NullContextChanger),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='reacher_hard', type=str, help='Environment name', required=False)
    parser.add_argument('--video-dir', default=None, type=str, help='Video dir', required=False)
    parser.add_argument('--episode_len', default=50, type=int, help='Video length', required=False)
    parser.add_argument('--im-w', default=64, type=int, help='Frame width', required=False)
    parser.add_argument('--im-h', default=64, type=int, help='Frame height', required=False)
    parser.add_argument('--num-train', default=5000, type=int, help='Num training videos', required=False)

    args, _ = parser.parse_known_args(sys.argv[1:])

    episode_len = args.episode_len
    task_name = args.env
    im_w, im_h = args.im_w, args.im_h
    env_name, expert_file, xml_file, cc_class = env_data[task_name]

    expert = DrQV2Agent.load(expert_file)
    expert.train(training=False)

    env = dmc.make(env_name, frame_stack=3, action_repeat=2, seed=2, episode_len=episode_len, xml_path=xml_file)
    video_dir = Path(f'videos/{args.video_dir if args.video_dir is not None else task_name }')
    utils.generate_video_from_expert(
        video_dir / 'exp/0', expert, env, cc_class(), num=args.num_train, im_w=im_w, im_h=im_h)
