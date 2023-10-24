import argparse
import os
import json

def get_args_parser():
    parser = argparse.ArgumentParser(description='TimesNet', add_help=False)
    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str,
                        default='test', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode', default=False)
    parser.add_argument('--use_wandb', type=str, default='wandb_runtime')
    parser.add_argument('--trace_freq', type=int, default=100)
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--fourcast_step', type=int, default=10)
    parser.add_argument('--time_step', type=int, default=2)
    parser.add_argument('--compute_graph_set', type=str, default=None)
    parser.add_argument('--pretrain_weight', type=str, default=None)
    parser.add_argument('--do_early_stop', type=int, default=1)
    parser.add_argument('--weight_name', type=str, default=None)
    
    # data loader
    parser.add_argument('--data', type=str,
                        default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str,
                        default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str,
                        default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str,
                        default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96,
                        help='input sequence length')
    parser.add_argument('--label_len', type=int,
                        default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str,
                        default='Monthly', help='subset for M4')

    # inputation task
    parser.add_argument('--mask_rate', type=float,
                        default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float,
                        default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int,
                        default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7,
                        help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7,
                        help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str,
                        default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int,
                        default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int,
                        default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size of train input data')
    parser.add_argument('--valid_batch_size', type=int,
                        default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3,
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help='optimizer learning rate')
    parser.add_argument('--accumulation_steps', type=int,
                        default=1, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--loss', type=str, default='MSE',
                        help='loss function')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true',
                        help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str,
                        default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int,
                        default=2, help='number of hidden layers in projector')
    parser.add_argument('--save_numpy_result', action='store_true',
                        help='save_numpy_resultQ', default=False)
    parser.add_argument('--monitor_alpha', action='store_true',
                        help='monitor_alpha', default=False)     
    parser.add_argument('--seed', type=int,
                        default=2021, help='seed')              
    return parser


def get_args(config_path=None):
    conf_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    conf_parser.add_argument(
        "-c", "--conf_file",  default=None, help="Specify config file", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args()
    defaults = {}
    if config_path:
        
        with open(config_path, 'r') as f:
            defaults = json.load(f)
    if args.conf_file:
        with open(args.conf_file, 'r') as f:
            defaults = json.load(f)
    # parser = argparse.ArgumentParser(parents=[conf_parser])
    parser = argparse.ArgumentParser(
        'GFNet training and evaluation script', parents=[get_args_parser()])
    parser.set_defaults(**defaults)

    config = parser.parse_known_args(remaining_argv)[0]
    config.config_file = args.conf_file
    return config


def heavily_main(args):
    import train
    train.main(args)


def check_exist_via_lock(file_path):
    # lock_dir = "/nvme/zhangtianning/share/lock"
    # lock_file = os.path.join(lock_dir, "TSL/train",file_path.replace("/", "-"))+'.lock'

    # if not os.path.exists(lock_dir):
    #     # we will try to mount it
    #     os.system("sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 zhangtianning@jump.pjlab.org.cn:Default/10.140.52.118/zhangtianning_share ~/share")

    # if len(os.listdir(lock_dir)) == 0:
    #     raise NotImplementedError("dont find the lock file, exit.............")

    # if os.path.exists(lock_file):
    #     print(f"====> detected experiment {lock_file} is running in other machine,  ====> skip..............")
    #     return True
    # else:
    #     print(f"====> do not find lock file at {lock_file} , ===> continue..............")
    experiment_dir = os.path.join("checkpoints", file_path)
    if os.path.exists(experiment_dir):
        print(
            f"====> detected experiment {experiment_dir} is running, skip..............")
        return True
    #os.system(f"touch {lock_file}")
    return False


def no_repeat_benchmark():
    args = get_args()
    pretrain_flag = 'FF' if args.mode == "finetune" else "ft"
    FLAG = f"bs_{args.batch_size}"
    projectdir = f'TSL-{args.model_id.split("_")[0]}/{args.model}/{pretrain_flag}{args.features}.{args.seq_len}_{args.label_len}_{args.pred_len}'
    FLAG = f"bs_{args.batch_size}{args.compute_graph_set}"
    file_path = os.path.join(projectdir, FLAG)
    if args.mode not in ['fourcast', 'replicate'] and check_exist_via_lock(file_path):
        exit()
    heavily_main(args)


if __name__ == "__main__":
    no_repeat_benchmark()
