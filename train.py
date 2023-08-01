
import os
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
import random
import numpy as np
import torch.distributed as dist
import torch
import time,json
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

def distributed_initial(args):
    import os
    ngpus = ngpus_per_node = torch.cuda.device_count()
    args.world_size = -1
    args.dist_file = None
    args.rank = 0
    args.dist_backend = "nccl"
    args.multiprocessing_distributed = ngpus > 1
    args.ngpus_per_node = ngpus_per_node
    if not hasattr(args, 'train_set'):
        args.train_set = 'large'
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "54247")
    hosts = int(os.environ.get("WORLD_SIZE", "1"))  # number of nodes
    rank = int(os.environ.get("RANK", "0"))  # node id
    gpus = torch.cuda.device_count()  # gpus per node
    args.dist_url = f"tcp://{ip}:{port}"
    if args.world_size == -1 and "SLURM_NPROCS" in os.environ:
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank = int(os.environ["SLURM_PROCID"])
        jobid = os.environ["SLURM_JOBID"]

        hostfile = "dist_url." + jobid + ".txt"
        if args.dist_file is not None:
            args.dist_url = "file://{}.{}".format(
                os.path.realpath(args.dist_file), jobid)
        elif args.rank == 0:
            import socket
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            args.dist_url = "tcp://{}:{}".format(ip, port)
            # with open(hostfile, "w") as f:f.write(args.dist_url)
        else:
            import os
            import time
            while not os.path.exists(hostfile):
                time.sleep(1)
            with open(hostfile, "r") as f:
                args.dist_url = f.read()
        print("dist-url:{} at PROCID {} / {}".format(args.dist_url,
              args.rank, args.world_size))
    else:
        args.world_size = 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    return args


def main_worker(local_rank, ngpus_per_node, args):
    setting = args.setting
    args.local_rank = local_rank
    if args.mode in ['pretrain', 'finetune']:
        args.SAVE_PATH = os.path.join(args.checkpoints, setting)
    if args.debug:
        args.SAVE_PATH = './debug'
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + local_rank
        print(f"start init_process_group,backend={args.dist_backend}, init_method={args.dist_url},world_size={args.world_size}, rank={args.rank}")
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)


    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast
    
    exp = Exp(args)  # set experiments
    logsys = exp.logsys
    if args.mode in ['pretrain','finetune']:
        if "finetune" in args.mode:
            assert args.pretrain_weight
            
            best_weight_path = os.path.join(args.pretrain_weight)
            logsys.info(f'loading model from {best_weight_path}.......')
            state_dict = torch.load(best_weight_path, map_location='cpu')
            if "model" in state_dict:
                epoch = state_dict["epoch"]
                state_dict=state_dict["model"]
            exp.model.load_state_dict(state_dict)
            exp.model.to(exp.device)
            logsys.info('done!')
        logsys.info('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        logsys.info('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

    if "monitor_alpha" in args.mode:
        exp.plot_multi_step_alpha(steps=args.fourcast_step, step_length=min(args.pred_len,args.seq_len),
                                  weight_name=args.weight_name,monitor_alpha=args.monitor_alpha)
    else:
        logsys.info('>>>>>>>testing multistep: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        folder_path = os.path.join(exp.logsys.ckpt_root, 'results')
        exp.plot_multi_step_prediction(steps=args.fourcast_step, step_length=min(args.pred_len,args.seq_len),folder_path=folder_path,
                                    save_numpy_result=args.save_numpy_result, weight_name=args.weight_name,monitor_alpha=args.monitor_alpha)
    # else:
    #     logsys.info('detect pred_len <= seq_len, skip....')
    exp.logsys.close()


def check_exist_via_lock(file_path, unique_flag=None, trail_limit=1):

    if unique_flag is not None:
        lock_dir = "/nvme/zhangtianning/share/lock"
        lock_file = os.path.join(lock_dir, "TSL/train",file_path.replace("/", "-"))+f'.lock.{unique_flag}'

        # if not os.path.exists(lock_dir):
        #     # we will try to mount it
        #     os.system("sshfs -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 zhangtianning@jump.pjlab.org.cn:Default/10.140.52.118/zhangtianning_share ~/share")
        
        # if len(os.listdir(lock_dir)) == 0:
        #     raise NotImplementedError("dont find the lock file, exit.............")


        if os.path.exists(lock_file):
            print(f"====> detected experiment {lock_file} is running in other machine,  ====> skip..............")
            return True
        else:
            print(f"====> do not find lock file at {lock_file} , ===> continue..............")
            os.system(f"touch {lock_file}")

    experiment_dir = os.path.join("checkpoints", file_path)
    if os.path.exists(experiment_dir):
        num_trails = len(os.listdir(experiment_dir))
        print(f"====> detected experiment {experiment_dir} has {num_trails} trails")
        if num_trails >= trail_limit:
            print(f"====> No.trails={num_trails} >= {trail_limit}, skip..............")
            return True
        else:
            print(f"====> No.trails={num_trails} < {trail_limit}, continue..............")
    return False
    

def get_file_path(args):
    pretrain_flag = 'TT' if args.mode =="finetune" else "ft"
    FLAG = f"bs_{args.batch_size}{args.compute_graph_set}"
    projectdir = f'TSL-{args.model_id.split("_")[0]}/{args.model}/{pretrain_flag}{args.features}.{args.seq_len}_{args.label_len}_{args.pred_len}'
    file_path = os.path.join(projectdir, FLAG)
    return file_path

def main(args=None):

    if args is None:
        import run
        args = run.get_args()
    args.use_gpu = True
    
    # fix_seed = args.seed
    # random.seed(fix_seed)
    # torch.manual_seed(fix_seed)
    # np.random.seed(fix_seed) # we claim at begin. and we verify it works.
    file_path = get_file_path(args)
    #if 'fourcast' not in args.mode and check_exist_via_lock(file_path):exit()
    #raise
    # os.makedirs(experiment_dir)
    args.seed = fix_seed
    TIME_NOW = time.strftime("%m_%d_%H_%M_%S")
    TIME_NOW = f"{TIME_NOW}-seed_{fix_seed}"
    setting = os.path.join(file_path, TIME_NOW)
    args.setting = setting
    args = distributed_initial(args)
    if args.multiprocessing_distributed:
        print("======== entering  multiprocessing train ==========")
        args.world_size = args.ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(
            main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        print("======== entering  single gpu train ==========")
        main_worker(0, args.ngpus_per_node, args)


if __name__ == "__main__":
    main()
