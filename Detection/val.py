import argparse
from yolov5.val import *
from my_utils import clean_labels


def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset.yaml', help='dataset.yaml path')
    parser.add_argument('--name', default='transmission_towers', help='save to project/name')
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    clean_labels()

    my_opt = my_parser()
    opt = parse_opt()

    opt.data = my_opt.data
    opt.name = my_opt.name

    main(opt)
