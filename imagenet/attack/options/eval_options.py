from .base_options import BaseOptions
import ast

class EvalOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--model_list", type=str, default='inceptionv3', help='specify use which model to evaluate')
        self.parser.add_argument("--csv_file", type=str, default='vgg11_attack.csv', help='specify use which csv file to write')
        self.parser.add_argument("--eval_metric", type=str, default='cross_entropy', choices=['cross_entropy', 'cosine_distance'], help='specify use which csv file to write')