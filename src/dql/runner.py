

from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dql.environment import Environment
from dql.logger import Logger
from dql.model import Model
from dql.trainer import Trainer
from dql.evaluator import Evaluator



def parse_argument():
    parser = ArgumentParser('My Implementation of deep q network',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--envname', type=str, nargs='?', default='CartPole-v0',
                        help='Need to be a valid gym atari environment')
    parser.add_argument('--play', action='store_true', help='flag the eval mode')
    parser.add_argument('-b', '--batch_size', type=int, nargs='?', default=32,
                        help='How many samples to gather from replay memory for each update')
    parser.add_argument('-n', '--num_episode', type=int, nargs='?', default=100,
                        help='number of episode to train on, this does not include the episode generated while populate replay memory')
    parser.add_argument('--frame_length', type=int, nargs='?', default=4,
                        help='this amount of frame with be stacked together to create a state')
    parser.add_argument('--replay_mem_cap', type=int, nargs='?', default=100,
                        help='this argument defines the capacity of replay memory, the number must be strictly greater than the size of batch')
    parser.add_argument('--gamma', type=float, nargs='?',
                        default=0.99, help='discount factor')
    parser.add_argument('--target_update', type=int, nargs='?', default=5,
                        help='This argument specify how frequent target net will be updated')
    parser.add_argument('--log_interval', type=int, nargs='?', default=1,
                        help='log will be written every <interval> episodes')
    parser.add_argument('--checkpoint_interval',
                        type=int, nargs='?', default=5, help='checkpoint of model will be written every <interval> episodes. This value also defines how frequently model will be evaluated during training')
    parser.add_argument('--log_dir', type=Path, nargs='?', default=Path('./runs'),
                        help='tensorboard events and saved model will be placed in <log_dir>/<envname> directory ')
    parser.add_argument('--eps_schedule', type=float, nargs='*', default=[
                        0.9, 0.05, 200], help='consume 3 values which defines the epsilon decay schedule (start, end, steps)')
    parser.add_argument('--image_size', type=int, nargs='*',
                        default=[84, 84], help='Size of cropped image')
    parser.add_argument('--eval_episode', type=int, nargs='?', default=10, help='how many episode to run while evaluating model. The reward is caluated as the average of those episode')
    
    parser.add_argument('--resume', type=bool, nargs='?', default=True, help='Resuem from the checkpoints from <log_dir>/runs/<envname>/checkpoints')

    args = parser.parse_args()
    if args.batch_size >= args.replay_mem_cap:
        raise ValueError(
            'Capacity of replay memory must be strictly greater than batch size. Otherwise how would you sample from it?')
    if len(args.eps_schedule) != 3:
        raise ValueError(
            'epsilon schedule must consume 3 values (start, end, step)')
    assert len(args.image_size) == 2
    return args


def main():
    args = parse_argument()
    hps = vars(args)
    env = Environment(hps)
    model = Model(env, hps)
    logger = Logger(hps)
    evaluator = Evaluator(env, model, hps, logger)
    trainer = Trainer(env, model, hps, logger, evaluator)
    
    checkpoint_path = (args.log_dir / args.envname / 'checkpoints').resolve()
    # print(checkpoint_dir)
    if args.resume and len(list(checkpoint_path.rglob('saved_model.ckpt-*'))) != 0:
        
        latest_episode_num = model.load_checkpoint()
        print('load checkpoint from {0}/{1}'.format(checkpoint_path,
                                                    'saved_model.ckpt-'+str(latest_episode_num)))
        logger.set_episode_offset(latest_episode_num)
    # if args.resume:
    #     model_ckpt_path = get_latest_model_path(hps)
    #     model.l
    if args.play:
        print('Start playing')
        evaluator.play()
    else:
        print('Start training')
        trainer.train()
    env.close()
    return


if __name__ == "__main__":
    main()
