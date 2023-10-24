# #!/bin/bash

# # 获取文件夹路径
# folder_path="scripts/long_term_forecast/"
# # 遍历文件夹下的所有文件
# find "$folder_path" -type f | while read file_path; do
#     # 删除包含CUDA_VISIBLE_DEVICES的行
#     sed -i '/CUDA_VISIBLE_DEVICES/d' "$file_path"
# done
lr=0.00001
python batch_run.py checkpoints fwd2_D_Rog5 $lr 3 1
python batch_run.py checkpoints fwd2_D $lr 3 1
# python test.py checkpoints fwd3_ABC_Log 0.00001 4
# python test.py checkpoints fwd3_ABC 0.00001 4
# python test.py checkpoints fwd2_D_Rog2 0.00001 3 
# python test.py checkpoints fwd3_KAR 0.00001 4
# python test.py checkpoints fwd3_D 0.00001 4
# python test.py checkpoints fwd3_ABC 0.00001 4
# python test.py checkpoints fwd3_ABC_Log 0.00001 4
