from .base_options import BaseOptions
import ast

class AttackOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--eps", type=float, default=16, help='specify momentum')
        self.parser.add_argument("--arch", type=str, default='inceptionv3', help='specify attack which model')
        self.parser.add_argument("--path", type=str, help='model path')
        self.parser.add_argument("--momentum", type=float, default=1.0, help='specify momentum')
        self.parser.add_argument("--num_iter", type=int, default=10, help='specify momentum')
        self.parser.add_argument("--alpha", type=float, default=2, help='specify step size')
        self.parser.add_argument("--attack_method", type=str, default='MI-FGSM', help='specify attack method')
        # loss function
        self.parser.add_argument("--loss_function", type=str, default='CE', choices=['CE'])
        # input_diversity
        self.parser.add_argument("--ensemble_num", type=int, default=1, help='specify ensemble iterations')
        self.parser.add_argument("--diversity_prob", type=float, default=0, help='specify input_diversity operation')
        self.parser.add_argument("--rotate_std", type=float, default=0.5, help='specify input_diversity operation')
        self.parser.add_argument("--pow_base", type=float, default=2, help='specify the base of pow')
        self.parser.add_argument("--m2_admix", type=int, default=0, help='specify the m2 for Admix')
        self.parser.add_argument("--eta_admix", type=float, default=0.2, help='specify the eta for Admix')
        self.parser.add_argument("--kernel_size", type=int, default=5, help='specify translation invariance kernel size')
        self.parser.add_argument("--diversity_method", type=str, default=None, help='specifty use which attack method')
        # attack method