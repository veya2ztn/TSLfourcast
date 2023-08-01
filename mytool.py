from ast import Not
from distutils.log import info
import os

import numpy as np
import argparse,tqdm,json,re
import sys
import pandas as pd

def remove_weight(trial_path):
    for file_name in ['pretrain_latest.pt','backbone.best.pt']:
        if  file_name in os.listdir(trial_path):   
            raise
            weight_path = os.path.join(trial_path,file_name)
            if os.path.islink(weight_path):continue
            #if not os.path.islink(weight_path):
            ceph_path   = os.path.join("~/cephdrive/FourCastNet",weight_path)
            os.system(f"rm {weight_path}")
            #time.sleep(0.4)
            os.system(f"ln -s {ceph_path} {weight_path}")

def assign_trail_job(trial_path,wandb_id=None, gpu=0):
    if "/ft" not in trial_path:return
    if "ETTm2 " in trial_path:return
    from tbparse import SummaryReader
    import wandb
    from exp.exp_basic import create_logsys
    from run import get_args

    args              = get_args(os.path.join(trial_path,"config.json"))
    if args is None:return
    args.use_wandb    = 'wandb_runtime'
    args.wandb_id     = wandb_id
    #args.wandb_resume = 'must'
    args.gpu          = gpu
    args.recorder_list = []
    logsys = create_logsys(args, save_config=False)
    epoch_pool  = {}
    test_pool   = {}
    iter_metric = []
    hparams1    = hparams2=None
    runtime_loss = {}
    longrange_prediction={}
    summary_dir = trial_path
    for filename in os.listdir(summary_dir):
        if 'event' not in filename:continue
        log_dir = os.path.join(summary_dir,filename)
        reader = SummaryReader(log_dir)
        df = reader.scalars
        if 'tag' not in reader.hparams:
            print(f"no hparams at {log_dir}")
            #print(reader.hparams)

        if len(df) < 1: 
            print(f"no scalars at {log_dir},pass")
            continue
        print("start parsing tensorboard..............")
        for key in set(df['tag'].values):
            if key !='longrange_prediction':continue
            all_pool = epoch_pool
            if 'tag' not in reader.hparams:
                hparams2={}
            else:
                hparams2 = dict([(name,v) for name,v in zip(reader.hparams['tag'].values,reader.hparams['value'].values)])
            now   = df[df['tag'] == key]
            steps = now['step'].values
            values= now['value'].values
            if key not in ['longrange_prediction','runtime_loss']:
                for step, val in zip(steps,values):
                    if step not in all_pool:all_pool[step]={}
                    all_pool[step][key]=val
            elif key == 'longrange_prediction':
                for step, val in zip(steps,values):
                    longrange_prediction[step] = val
            elif key == 'runtime_loss':
                for step, val in zip(steps,values):
                    runtime_loss[step] = val
    print("tensorboard parse done, start wandb..............")
    # hparams = hparams2 if hparams1 is None else hparams1
    # if hparams == None:return
    # with open(os.path.join(trial_path,'config.json'), 'r') as f:
    #     hparams = json.load(f)
    # with open(os.path.join(trial_path,"results/result.txt")) as f:
    #     results = [line for line in f if 'mse' in line ][0].strip()
    #     mse,mae = results.split(',')
    #     mse     = float(mse.split(':')[-1])
    #     mae     = float(mae.split(':')[-1])
    #     print(mse,mae)
    # logsys.wandblog({"test_mae": mae, "test_mse": mse})
    for epoch, value_pool in epoch_pool.items():
        for key, val in value_pool.items():
            logsys.record(key, float(val), epoch, epoch_flag="epoch")  
    for predict_distance, error in longrange_prediction.items():
        logsys.record("longrange_prediction", float(error), predict_distance, epoch_flag="future")
    for _iter, loss in runtime_loss.items():
        logsys.record("runtime_loss", float(loss), _iter, epoch_flag="iter")
    logsys.close()
    print("all done..............")

def run_finetune(path):
    from run import get_args
    from train import main_worker, check_exist_via_lock, main, get_file_path
    if ("TSL-ili" in path or 
        "traffic" in path or
        "Exchange" in path or
        "ECL" in path or 
        "ETSformer" in path or 
        "/ft" not in path or
        "_96/" not in path
        ):return

    # print(path)
    args = run.get_args(os.path.join(path, "config.json"))
    args.mode = "finetune"
    args.pretrain_weight = os.path.join(path, "checkpoint.pth")
    args.valid_batch_size = 64
    args.fourcast_step = 10
    print(args.learning_rate)
    print(args.compute_graph_set)
    print(args.time_step)
    raise
    # args.learning_rate = float(lr)
    # args.compute_graph_set = compute_graph_set
    # args.time_step = time_step
    # pretrain_flag = 'FF' if args.mode == "finetune" else "ft"
    # FLAG = f"bs_{args.batch_size}{args.compute_graph_set}"
    # projectdir = f'TSL-{args.model_id.split("_")[0]}/{args.model}/{pretrain_flag}{args.features}.{args.seq_len}_{args.label_len}_{args.pred_len}'
    # file_path = os.path.join(projectdir, FLAG)
    file_path = get_file_path(args)
    if check_exist_via_lock(file_path):
        return 

    main(args)


def run_fourcast(ckpt_path,step = 4*24//6,force_fourcast=False,wandb_id=None,weight_chose=None):
    from run import get_args
    from train import main_worker, check_exist_via_lock, main, get_file_path
    result_path = os.path.join(ckpt_path, "result")
    if os.path.exists(result_path) and not force_fourcast:return
    args = run.get_args(os.path.join(ckpt_path, "config.json"))
    args.mode = "fourcast"
    main(args)

def remove_trail_path(trial_path):
    trail_file_list= os.listdir(trial_path)
    if ('seed' in trial_path.split('/')[-1] and
        'checkpoint.pth' not in trail_file_list and
        'results' not in trail_file_list and
            'test_results' not in trail_file_list):
        #os.system(f"ls {trail_path}")
        os.system(f"rm -rf {trail_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('parse tf.event file to wandb', add_help=False)
    parser.add_argument('--paths',type=str,default="")
    parser.add_argument('--moded',type=str,default="dryrun")
    parser.add_argument('--level', default=1, type=int)
    parser.add_argument('--divide', default=1, type=int)
    parser.add_argument('--part', default=0, type=int)
    parser.add_argument('--fourcast_step', default=4*24//6, type=int)
    parser.add_argument('--path_list_file',default="",type=str)
    parser.add_argument('--force_fourcast',default=0,type=int)
    parser.add_argument('--weight_chose',default=None,type=str)
    args = parser.parse_known_args()[0]

    if args.paths != "":
        level = args.level
        root_path = args.paths
        now_path = [root_path]
        while level>0:
            new_path = []
            for root_path in now_path:
                if os.path.isfile(root_path):continue
                if len(os.listdir(root_path))==0:
                    os.system(f"rm -r {root_path}")
                    continue
                for sub_name in os.listdir(root_path):
                    sub_path =  os.path.join(root_path,sub_name)
                    if os.path.isfile(sub_path):continue
                    new_path.append(sub_path)
            now_path = new_path
            level -= 1
    
    now_path_pool = None
    if 'json' in args.path_list_file:
        with open(args.path_list_file, 'r') as f:
            path_list_file = json.load(f)
        if isinstance(path_list_file,dict):
            now_path_pool = path_list_file
            now_path = list(now_path_pool.keys())
        else:
            now_path = path_list_file

    print(f"we detect {len(now_path)} trail path; \n  from {now_path[0]} to \n  {now_path[-1]}")
    total_lenght = len(now_path)
    length = int(np.ceil(1.0*total_lenght/args.divide))
    s    = int(args.part)
    now_path = now_path[s*length:(s+1)*length]
    print(f"we process:\n  from  from {now_path[0]}\n to  {now_path[-1]}")

    
    if args.moded == 'dryrun':exit()
    for trail_path in tqdm.tqdm(now_path):
        trail_path = trail_path.strip("/")
        if len(os.listdir(trail_path))==0:
            os.system(f"rm -r {trail_path}")
            continue
        #os.system(f"sensesync sync s3://QNL1575AXM6DF9QUZDA9:BiulXCfnNpIx6tl6P14I8W5QDw6NSU3yqSWVdbkH@FourCastNet.10.140.2.204:80/{trail_path}/ {trail_path}/")
        #os.system(f"sensesync sync {trail_path}/ s3://QNL1575AXM6DF9QUZDA9:BiulXCfnNpIx6tl6P14I8W5QDw6NSU3yqSWVdbkH@FourCastNet.10.140.2.204:80/{trail_path}/ ")
        #os.system(f"aws s3 --endpoint-url=http://10.140.2.204:80 --profile zhangtianning sync s3://FourCastNet/{trail_path}/ {trail_path}/")
        # print(trail_path)
        # print(os.listdir(trail_path))
        if   args.moded == 'fourcast':run_fourcast(trail_path,step=args.fourcast_step,force_fourcast=args.force_fourcast,weight_chose=args.weight_chose)
        elif args.moded == 'tb2wandb':assign_trail_job(trail_path)
        elif args.moded == 'cleantmp':remove_trail_path(trail_path)
        elif args.moded == 'cleanwgt':remove_weight(trail_path)
        elif args.moded == 'createtb':create_fourcast_table(trail_path,force_fourcast=args.force_fourcast)
        elif args.moded == 'snap_nodal':run_snap_nodal(trail_path,step=args.fourcast_step,force_fourcast=args.force_fourcast,weight_chose=args.weight_chose)
        elif args.moded == 'createtb_nodalsnap':create_nodalsnap_table(trail_path)
        elif args.moded == 'createmultitb':create_multi_fourcast_table(trail_path,force=args.force_fourcast)
        else:
            raise NotImplementedError
