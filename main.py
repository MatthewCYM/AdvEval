from argparse import ArgumentParser
from transformers import set_seed
from src import evaluator_list, task_list, optimizer_list
import os
import pickle as pkl
import google.generativeai as palm


palm.configure(api_key=os.environ["PALM_API_KEY"])


def read_args():
    parser = ArgumentParser()

    # evaluator
    parser.add_argument('--gold_evaluator_name', type=str, default='palm_dialog')
    parser.add_argument('--victim_evaluator_name', type=str, default='poe')

    # optimizer
    parser.add_argument('--optimizer_name', type=str, default='optimizer_dialog_palm')

    # task
    parser.add_argument('--task_name', type=str, default='response_dailydialog')
    parser.add_argument('--use_train_data', action='store_true')
    parser.add_argument('--demo_size', type=int, default=2)
    parser.add_argument('--demo_start', type=int, default=0)

    parser.add_argument('--seed', type=int, default=42)

    known_args, _ = parser.parse_known_args()
    assert known_args.gold_evaluator_name != known_args.victim_evaluator_name

    # add model/task specific arguments
    gold_evaluator_cls = evaluator_list[known_args.gold_evaluator_name]
    gold_evaluator_cls.add_args(parser)

    victim_evaluator_cls = evaluator_list[known_args.victim_evaluator_name]
    victim_evaluator_cls.add_args(parser)

    optimizer_cls = optimizer_list[known_args.optimizer_name]
    optimizer_cls.add_args(parser)

    task_cls = task_list[known_args.task_name]
    parser = task_cls.add_args(parser)

    args = parser.parse_args()
    return args


def main():
    args = read_args()

    set_seed(args.seed)

    gold_evaluator = evaluator_list[args.gold_evaluator_name](args)
    victim_evaluator = evaluator_list[args.victim_evaluator_name](args)

    task = task_list[args.task_name](args, [gold_evaluator, victim_evaluator])
    optimizer = optimizer_list[args.optimizer_name](args, task, gold_evaluator, victim_evaluator)

    out_file = f'{gold_evaluator.get_name()}-{victim_evaluator.get_name()}-{optimizer.get_name()}-{task.get_name()}.pkl'

    rtn = optimizer.evaluate()

    with open(out_file, 'wb') as f:
        pkl.dump(rtn, f)


if __name__ == '__main__':
    main()
