import argparse
import random
import numpy as np
import torch
from utils.utils import  boolean_string
from model_train import single_run_entrance


def main(args):
    print(args)
    # set up seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    args.trick = {'labels_trick': args.labels_trick, 'separated_softmax': args.separated_softmax,
                  'kd_trick': args.kd_trick, 'kd_trick_star': args.kd_trick_star, 'review_trick': args.review_trick,
                  'ncm_trick': args.ncm_trick}
    args.epoch = args.epochs
    args.batch = args.batch_size
    args.test_batch = args.test_batch_size
    single_run_entrance(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ####################################################################
    # Parse command line
    ####################################################################
    parser.add_argument('--training_set', type=str, default='VFHQ', nargs='+',
                        help='Training dataset select from: VFHQ and ForenSynths')
    parser.add_argument('--test_set', type=str, default='VFHQ', nargs='+', help='')
    parser.add_argument('--feature', default='fft', help='Feature used for training, choose from image and fft')
    parser.add_argument('--mode', type=int, default=0,
                        help='fft frequency band, 0: full, 1: low, 2: mid, 3: high')

    parser.add_argument('--dataroot', type=str, default='./datasets/', help='path to cached dataset')
    parser.add_argument('--continual_learning', default='NoCL',
                        help='Feature used for training, choose from NoCL , Normal and CL')
    parser.add_argument('--jpg_level', type=str, default='90',
                        help='Test with different jpg compression effiecients, only effective when use jpg for test set.')
    parser.add_argument('--resize_size', type=str, default='200',
                        help='Test with different resize sizes, only effective when use resize for test set.')

    parser.add_argument('--enable-logging', type=bool, default=False,
                        help='output to tensorlogger')
    parser.add_argument('--log-dir', default='./log/',
                        help='folder to output log')
    parser.add_argument('--store', type=boolean_string, default=False,
                            help='Store result or not (default: %(default)s)')
    parser.add_argument('--model_dir', default='./model/',
                        help='folder to output model checkpoints')
    parser.add_argument('--model', default='resnet',
                        help='Base classification model')
    parser.add_argument('--num-workers', default=1,
                        help='Number of workers to be created')
    parser.add_argument('--pin-memory', type=bool, default=True,
                        help='')
    parser.add_argument('--resume', default=False, type=bool,
                        help='resume last models latest checkpoint')
    parser.add_argument('--resume_file', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=1, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=10,
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-decay', default=1e-2, type=float,
                        help='learning rate decay ratio (default: 1e-6')
    parser.add_argument('--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--check_cached', action='store_true', default=False,
                        help='Use cached dataset or not')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--interval', type=int, default=5,
                        help='logging interval, epoch based. (default: 5)')

    # Device options
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    # Continual Learning
    parser.add_argument('--buffer_tracker', type=boolean_string, default=False,
                            help='Keep track of buffer with a dictionary')

    ########################General#########################
    parser.add_argument('--num_runs', dest='num_runs', default=1, type=int,
                            help='Number of runs (default: %(default)s)')

    ########################Misc#########################
    parser.add_argument('--val_size', dest='val_size', default=0, type=float,
                            help='val_size (default: %(default)s)')
    parser.add_argument('--num_val', dest='num_val', default=3, type=int,
                            help='Number of batches used for validation (default: %(default)s)')
    parser.add_argument('--num_runs_val', dest='num_runs_val', default=3, type=int,
                            help='Number of runs for validation (default: %(default)s)')
    parser.add_argument('--error_analysis', dest='error_analysis', default=False, type=boolean_string,
                            help='Perform error analysis (default: %(default)s)')
    parser.add_argument('--verbose', type=boolean_string, default=True,
                            help='print information or not (default: %(default)s)')


    ########################Data#########################
    parser.add_argument('--num_tasks', dest='num_tasks', default=10,
                            type=int,
                            help='Number of tasks (default: %(default)s), OpenLORIS num_tasks is predefined')
    parser.add_argument('--fix_order', dest='fix_order', default=False,
                            type=boolean_string,
                            help='In NC scenario, should the class order be fixed (default: %(default)s)')
    parser.add_argument('--plot_sample', dest='plot_sample', default=False,
                            type=boolean_string,
                            help='In NI scenario, should sample images be plotted (default: %(default)s)')
    parser.add_argument('--data', dest='data', default="VFHQ",
                            help='Path to the dataset. (default: %(default)s)')
    parser.add_argument('--cl_type', dest='cl_type', default="ni", choices=['nc', 'ni'],
                            help='Continual learning type: new class "nc" or new instance "ni". (default: %(default)s)')
    parser.add_argument('--ns_factor', dest='ns_factor', nargs='+',
                            default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), type=float,
                            help='Change factor for non-stationary data(default: %(default)s)')
    parser.add_argument('--ns_type', dest='ns_type', default='noise', type=str, choices=['noise', 'occlusion', 'blur'],
                            help='Type of non-stationary (default: %(default)s)')
    parser.add_argument('--ns_task', dest='ns_task', nargs='+', default=(1, 1, 2, 2, 2, 2), type=int,
                            help='NI Non Stationary task composition (default: %(default)s)')
    parser.add_argument('--online', dest='online', default=True,
                            type=boolean_string,
                            help='If False, offline training will be performed (default: %(default)s)')

    ########################Agent#########################
    parser.add_argument('--agent', dest='agent', default='ER',
                            choices=['ER', 'EWC', 'AGEM', 'CNDPM', 'LWF', 'ICARL', 'GDUMB', 'ASER', 'SCR'],
                            help='Agent selection  (default: %(default)s)')
    parser.add_argument('--update', dest='update', default='random', choices=['random', 'GSS', 'ASER'],
                            help='Update method  (default: %(default)s)')
    parser.add_argument('--retrieve', dest='retrieve', default='random', choices=['MIR', 'random', 'ASER', 'match', 'mem_match'],
                            help='Retrieve method  (default: %(default)s)')

    #######################Tricks#########################
    parser.add_argument('--labels_trick', dest='labels_trick', default=False, type=boolean_string,
                            help='Labels trick')
    parser.add_argument('--separated_softmax', dest='separated_softmax', default=False, type=boolean_string,
                            help='separated softmax')
    parser.add_argument('--kd_trick', dest='kd_trick', default=False, type=boolean_string,
                            help='Knowledge distillation with cross entropy trick')
    parser.add_argument('--kd_trick_star', dest='kd_trick_star', default=False, type=boolean_string,
                            help='Improved knowledge distillation trick')
    parser.add_argument('--review_trick', dest='review_trick', default=False, type=boolean_string,
                            help='Review trick')
    parser.add_argument('--ncm_trick', dest='ncm_trick', default=False, type=boolean_string,
                            help='Use nearest class mean classifier')
    parser.add_argument('--mem_iters', dest='mem_iters', default=1, type=int,
                            help='mem_iters')

    ########################Optimizer#########################
    parser.add_argument('--optimizer', dest='optimizer', default='SGD', choices=['SGD', 'Adam'],
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.1,
                        type=float,
                        help='Learning_rate (default: %(default)s)')
    parser.add_argument('--epoch', dest='epoch', default=1,
                        type=int,
                        help='The number of epochs used for one task. (default: %(default)s)')
    parser.add_argument('--batch', dest='batch', default=10,
                        type=int,
                        help='Batch size (default: %(default)s)')
    parser.add_argument('--test_batch', dest='test_batch', default=128,
                        type=int,
                        help='Test batch size (default: %(default)s)')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0,
                        help='weight_decay')

    ########################GDumb#########################
    parser.add_argument('--minlr', dest='minlr', default=0.0005, type=float, help='Minimal learning rate')
    parser.add_argument('--clip', dest='clip', default=10., type=float,
                    help='value for gradient clipping')
    parser.add_argument('--mem_epoch', dest='mem_epoch', default=70, type=int, help='Epochs to train for memory')

    ########################ER#########################
    parser.add_argument('--mem_size', dest='mem_size', default=5000,
                        type=int,
                        help='Memory buffer size (default: %(default)s)')
    parser.add_argument('--eps_mem_batch', dest='eps_mem_batch', default=10,
                        type=int,
                        help='Episode memory per batch (default: %(default)s)')

    ########################MIR#########################
    parser.add_argument('--subsample', dest='subsample', default=50,
                        type=int,
                        help='Number of subsample to perform MIR(default: %(default)s)')

    ########################TEST#########################
    parser.add_argument('--runtest', dest='runtest', default=False,
                        type=bool,
                        help='Is run for test, default: %(default)s)')

    parser.add_argument('--loadmodel', dest='loadmodel', default='./model/NoCL_VFHQ_fft_0_resnet_checkpoint_latest.pth',
                        type=str,
                        help='load saved model from (default: %(default)s)')

    parser.add_argument('--runtestall', dest='runtestall', default=False,
                        type=bool,
                        help='Is run test for all model, default: %(default)s)')


    m_args = parser.parse_args()
    m_args.cuda = not m_args.no_cuda and torch.cuda.is_available()
    main(m_args)



