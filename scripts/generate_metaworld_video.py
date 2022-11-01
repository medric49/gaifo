import argparse
import random
import sys

import context_changers
import drqv2
import utils
from pathlib import Path
import dmc
from metaworld.policies.sawyer_button_press_topdown_v2_policy import SawyerButtonPressTopdownV2Policy
from metaworld.policies.sawyer_button_press_v2_policy import SawyerButtonPressV2Policy
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
import metaworld.policies
import metaworld_env


env_data = {
    'window_close': ('exp_local/window_close/1/snapshot.pt', 'window-close-v2'),
    'door_open': ('exp_local/door_open/1/snapshot.pt', 'door-open-v2'),
    'button_press_topdown': (SawyerButtonPressTopdownV2Policy, 'button-press-topdown-v2'),
    'reach': (SawyerReachV2Policy, 'reach-v2'),
    'button_press': (SawyerButtonPressV2Policy, 'button-press-v2'),
    'plate_slide': (metaworld.policies.SawyerPlateSlideV2Policy, 'plate-slide-v2'),
    'drawer_close': (metaworld.policies.SawyerDrawerCloseV2Policy, 'drawer-close-v2'),
}


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def train(self, *args, **kwargs):
        pass

    def act(self, *args, **kwargs):
        return random.uniform(self.env.action_spec().minimum, self.env.action_spec().maximum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='window_close', type=str, help='Environment name', required=False)
    parser.add_argument('--video-dir', default=None, type=str, help='Video dir', required=False)
    parser.add_argument('--episode_len', default=50, type=int, help='Video length', required=False)
    parser.add_argument('--im-w', default=64, type=int, help='Frame width', required=False)
    parser.add_argument('--im-h', default=64, type=int, help='Frame height', required=False)
    parser.add_argument('--num-train', default=5000, type=int, help='Num training videos', required=False)
    args, _ = parser.parse_known_args(sys.argv[1:])

    env_dir = args.env
    im_w, im_h = args.im_w, args.im_h
    expert_file, env_name = env_data[env_dir]
    episode_len = args.episode_len

    video_dir = Path(f'videos/{args.video_dir if args.video_dir is not None else env_dir }')

    env = metaworld_env.Env(env_name)
    env = dmc.wrap(env, frame_stack=3, action_repeat=2, episode_len=episode_len)

    if type(expert_file) != str:
        policy = expert_file()
        expert = metaworld_env.Expert(policy, env)
    else:
        expert = drqv2.DrQV2Agent.load(expert_file)
    expert.train(False)

    utils.generate_video_from_expert(
        video_dir / 'exp/0', expert, env, context_changers.NullContextChanger(),
        num=args.num_train, im_w=im_w, im_h=im_h)
