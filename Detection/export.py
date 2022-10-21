import argparse
from yolov5.export import *
from my_utils import clean_labels


def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset.yaml', help='dataset.yaml path')
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    clean_labels()

    my_opt = my_parser()
    opt = parse_opt()

    opt.data = my_opt.data
    opt.include = ['onnx']

    main(opt)
