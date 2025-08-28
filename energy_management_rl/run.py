import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Energy Management RL Pipeline')
    parser.add_argument('--mode', type=str, choices=['test', 'train', 'inference', 'all'], 
                        default='all', help='Mode to run (test, train, inference, or all)')
    parser.add_argument('--timesteps', type=int, default=100000, 
                        help='Total timesteps to train for')
    parser.add_argument('--episodes', type=int, default=3, 
                        help='Number of episodes to run during inference')
    parser.add_argument('--render', action='store_true', 
                        help='Whether to render the environment during inference')
    
    args = parser.parse_args()
    
    if args.mode == 'test' or args.mode == 'all':
        print("Testing environment...")
        os.system('python test_env.py')
    
    if args.mode == 'train' or args.mode == 'all':
        print("Training agent...")
        os.system(f'python train.py --timesteps {args.timesteps}')
    
    if args.mode == 'inference' or args.mode == 'all':
        print("Running inference...")
        cmd = f'python inference.py --episodes {args.episodes}'
        if args.render:
            cmd += ' --render'
        os.system(cmd)


if __name__ == "__main__":
    main()
