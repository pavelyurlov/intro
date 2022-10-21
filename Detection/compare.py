import argparse
from yolov5.val import *
from my_utils import clean_labels


def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs=2, type=str, help='model paths, first torch, then onnx')
    parser.add_argument('--data', type=str, default='dataset.yaml', help='dataset.yaml path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--name', default='transmission_towers', help='save to project/name')
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    clean_labels()

    my_opt = my_parser()
    opt = parse_opt()

    # pytorch
    print('\n\nPYTORCH\n\n')

    opt.weights = my_opt.weights[:1]
    opt.data = my_opt.data
    opt.batch_size = my_opt.batch_size
    opt.name = my_opt.name + '_pytorch'

    main(opt)

    # onnx
    print('\n\nONNX\n\n')
    opt.weights = my_opt.weights[1:]
    opt.name = my_opt.name + '_onnx'

    main(opt)
