# import os 

# dataset_path = "checkpoints/TSL-custom"

# #dataset_path = os.path.join(ROOTDIR,dataset_name)
# for model_name in os.listdir(dataset_path): 
#     model_path = os.path.join(dataset_path,model_name)
#     for job_name in os.listdir(model_path): 
#         job_path = os.path.join(model_path,job_name)
#         for bs_name in os.listdir(job_path):
#             bs_path = os.path.join(job_path,bs_name)
#             for trail_name in os.listdir(bs_path):
#                 trail_path = os.path.join(bs_path, trail_name)
#                 if "result" not in os.listdir(trail_path):
#                     os.system(f"rm -r {trail_path}")
                    
import os
import sys

now_path = [sys.argv[1]]
compute_graph_set = sys.argv[2]
lr        = sys.argv[3]
time_step = int(sys.argv[4])
unique_flag = int(sys.argv[5])
#level=5
# while level>0:
#     new_path = []
#     for root_path in now_path:
#         if os.path.isfile(root_path):continue
#         if len(os.listdir(root_path))==0:
#             os.system(f"rm -r {root_path}")
#             continue
#         for sub_name in os.listdir(root_path):
#             sub_path =  os.path.join(root_path,sub_name)
#             if os.path.isfile(sub_path):continue
#             new_path.append(sub_path)
#     now_path = new_path
#     level -= 1
# import numpy as np
# print(f"we detect {len(now_path)} trail path; \n  from {now_path[0]} to \n  {now_path[-1]}")

with open("pretrain_ckpt_list",'r') as f:
    now_path = [line.strip() for line in f ]



from run import get_args
from train import main_worker,check_exist_via_lock,main,get_file_path
for path in now_path:

    if "checkpoint.pth" not in os.listdir(path):
        print(f"no weight in {path} !!!")
        #os.system(f"rm -r {path}")
        continue
    # if "results" in os.listdir(path):
    #     if np.any(["multistep" in n for n in os.listdir(os.path.join(path,"results"))]):
    #         continue
    if "TSL-ili" in path:continue
    if ("traffic" not in path) or ("weather" not in path):continue
    #if "ECL" in path:continue
    if "ETSformer" in path:continue
    if "_96/" not in path:continue
    if "/ft" not in path:continue
    #print(path)
    args = get_args(os.path.join(path,"config.json"))
    args.mode = "finetune"
    args.pretrain_weight = os.path.join(path,"checkpoint.pth")
    args.time_step = time_step
    args.valid_batch_size  = 64
    args.fourcast_step     = 10 
    args.compute_graph_set = compute_graph_set
    args.learning_rate     = float(lr)
    args.num_workers       = 4
    # pretrain_flag = 'FF' if args.mode == "finetune" else "ft"
    # FLAG = f"bs_{args.batch_size}{args.compute_graph_set}"
    # projectdir = f'TSL-{args.model_id.split("_")[0]}/{args.model}/{pretrain_flag}{args.features}.{args.seq_len}_{args.label_len}_{args.pred_len}'
    # file_path = os.path.join(projectdir, FLAG)
    file_path = get_file_path(args)
    if check_exist_via_lock(file_path, unique_flag=None, trail_limit=5):
        continue
    main(args)

