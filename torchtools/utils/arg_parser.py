import argparse
import ast
import platform
from time import strftime


class ArgParser:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Set the arguments for models, training schedulers etc.')
        self._init_data_args(parser)
        self._init_model_args(parser)
        self._init_train_args(parser)
        self._init_lrate_args(parser)
        self._init_control_args(parser)
        self._args = parser.parse_args()

        self._json_data = self._format_model()
        self._train_info = self._format_train()
        self._lrate_sched_info = self._format_scheduler()
        self._json_data["train_params"] = self._train_info
        self._train_info["lrate"] = self._lrate_sched_info

    @property
    def args(self):
        return self._args

    @staticmethod
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    @staticmethod
    def _init_data_args(parser):
        # Data Options
        data_args = parser.add_argument_group('Data arguments')
        # data_args.add_argument('--dataset', metavar='NAME', default='minc',
        #                        choices=['minc2500', 'minc', 'minc_zhao'],
        #                        help='name of the datasets to be used' +
        #                             ' (default: minc2500)')
        data_args.add_argument('--data-root', metavar='DIR', help='path to ' +
                                                                  'datasets (default: ./$(DATASET)_root)',
                               default='../../data/material/MINC/original-paper/')
        data_args.add_argument('--save-dir', metavar='DIR', default='./results',
                               help='path to trained models (default: results/)')
        data_args.add_argument('--train-ops', metavar='Bool', default="",
                               help='Train with opensurface dataset')
        data_args.add_argument('--train-ops-mat-only', metavar='Bool', default="",
                               help='Train with opensurface dataset, but with material label only')
        data_args.add_argument('--train-ops-with-boundary', metavar='Bool', default="",
                               help='Train with opensurface dataset, with boundary loss applied to mat segments')
        data_args.add_argument('--chk-dir', metavar='DIR', default='./checkpoints',
                               help='path to checkpoints (default: checkpoints/)')
        data_args.add_argument('--workers', metavar='NUM', type=int,
                               default=0, help='number of worker threads for' +
                                               ' the data loader')
        data_args.add_argument('--gpus', metavar='NUM', type=int,
                               default=0, help='number of gpus per node')
        data_args.add_argument('--num-nodes', metavar='NUM', type=int,
                               default=0, help='number of nodes')
        data_args.add_argument('--split', metavar='NUM', type=int,
                               default=1, help='which split to train')
        data_args.add_argument('--generate_pseudo', metavar='NUM', type=int,
                               default=0, help='if to generate pseudo or not')
        data_args.add_argument('--uncertainty', metavar='bool', default="",
                               help='retrain the last layer with random seeds')

    @staticmethod
    def _init_control_args(parser):
        # Other Options
        parser.add_argument('--stage', default='patch', choices=['patch', 'scene'],
                            help='train the patch classification or ' +
                                 'full scene segmentation task (default: patch)')
        parser.add_argument('--resume', default='', type=str, metavar='JSON_FILE',
                            help='resume the training from the specified JSON ' +
                                 'file  (default: none)')
        parser.add_argument('--test', default='', type=str, metavar='JSON_FILE',
                            help='test the network from the specified checkpoint')
        parser.add_argument('--pre_train', default='', type=str, metavar='JSON_FILE',
                            help='train the network from the specified checkpoint')
        parser.add_argument('--tag', default='ResNeSt',
                            help='The reference name of the experiment')
        parser.add_argument('--testindoor', default='', type=str, metavar='JSON_FILE',
                            help='test the network from the specified checkpoint with indoor images')
        parser.add_argument('--infer', default='', type=str, metavar='JSON_FILE',
                            help='infer the images inside a folder')
        parser.add_argument('--crf', default='', type=str, metavar='JSON_FILE',
                            help='enable convcrf after the network output')
        parser.add_argument('--crftest', default='', type=str, metavar='JSON_FILE',
                            help='enable convcrf after the network output')


    @staticmethod
    def _init_lrate_args(parser):
        # Learning Rate Scheduler Options
        lrate_args = parser.add_argument_group('Learning rate arguments')
        lrate_args.add_argument('--l-rate', type=float, default=0.1,
                                metavar='NUM', help='initial learning Rate' +
                                                    ' (default: 0.1)')
        lrate_args.add_argument('--lrate-sched-mode', default="multistep",
                                metavar="NAME", help="name of the learning " +
                                                     "rate scheduler (default: constant)",
                                choices=['step', 'multistep', 'exponential',
                                         'constant'])
        lrate_args.add_argument('--milestones', default='[5,10]', metavar='LIST',
                                help='epoch indices for learning rate reduction' +
                                     ' (multistep, default: [5,10])')
        lrate_args.add_argument('--gamma', type=float, default=0.1,
                                metavar='NUM', help='multiplicative factor of ' +
                                                    'learning rate decay (default: 0.1)')
        lrate_args.add_argument('--step-size', type=int, default=5,
                                metavar='NUM', help='pediod of learning rate ' +
                                                    'decay (step, default: 5)')

    def _init_train_args(self, parser):
        # Training Options
        train_args = parser.add_argument_group('Training arguments')
        train_args.add_argument('--method', default='SGD', metavar='NAME',
                                help='training method to be used')
        # train_args.add_argument('--gpu', type=int, default=device_count(), metavar='NUM',
        #                         help='number of GPUs to use')
        train_args.add_argument('--epochs', default=20, type=int, metavar='NUM',
                                help='number of total epochs to run (default: 20)')
        train_args.add_argument('-b', '--batch-size', default=128, type=int,
                                metavar='NUM',
                                help='mini-batch size (default: 64)')
        train_args.add_argument('--momentum', type=float, default=0.9,
                                metavar='NUM', help='Momentum (default: 0.9)')
        train_args.add_argument('--w-decay', type=float, default=1e-4,
                                metavar='NUM', help='weigth decay (default: 1e-4)')
        train_args.add_argument('--seed', type=int, metavar='NUM',
                                default=179424691,
                                help='random seed (default: 179424691)')
        train_args.add_argument('--debug', default=False, const=True, metavar='DEBUG', type=self._str2bool, nargs="?",
                                help='enable the debug mode (default: False)')
        train_args.add_argument('--mode', type=int, default=1,
                                metavar='NUM', help='The mode of training')

    @staticmethod
    def _init_model_args(parser):
        # Model Options
        model_args = parser.add_argument_group('Model arguments')
        model_args.add_argument('-m', '--models', metavar='NAME',
                                default='densenet121', type=str,
                                help='name of the pre-trained network models to be used')
        model_args.add_argument('--classes', metavar='NUM',
                                default='23', type=int,
                                help='number of the classes to be used for the' +
                                     ' classification')

    def _format_model(self):
        classes = self._args.classes
        stage = self._args.stage
        tag = self._args.tag

        return {"platform": platform.platform(),
                "date": strftime("%Y-%m-%d_%H:%M:%S"),
                "impl": "pytorch",
                "classes": classes,
                "stage": stage,
                "tag": tag
                }

    def _format_train(self):

        return {"method": self._args.method,
                "epochs": self._args.epochs,
                "batch_size": self._args.batch_size,
                "momentum": self._args.momentum,
                "w_decay": self._args.w_decay,
                "l_rate": self._args.l_rate,
                "train_time": 0.0
                }

    def _format_scheduler(self):
        return {"last_epoch": 0,
                "gamma": self._args.gamma,
                "lrate_sched_mode": self._args.lrate_sched_mode,
                "step_size": self._args.step_size,
                "milestones": ast.literal_eval(self._args.milestones)
                }

    @property
    def json_data(self):
        return self._json_data

    @property
    def train_info(self):
        return self._train_info

    @property
    def lrate_sched_info(self):
        return self._lrate_sched_info