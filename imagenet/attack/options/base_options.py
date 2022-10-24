import argparse
import os
import torch
import ast

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        self.parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
        self.parser.add_argument("--img_size", type=int, default=299, help="size of each image dimension")
        self.parser.add_argument("--input_dir", type=str, default='input/imagenet_dev', help='input dir')
        self.parser.add_argument("--save_dir", type=str, required=True, help='specify which directory')
        self.parser.add_argument("--img_num", type=int, default=1000, help='specify the number of images')
        self.parser.add_argument("--print_interval", type=int, default=50, help='specify the number of images')
        self.parser.add_argument("--targeted", type=ast.literal_eval, default=False, help='specify untargeted/targeted attack')
        self.parser.add_argument("--seed", type=int, default=11037, help='random seed')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
