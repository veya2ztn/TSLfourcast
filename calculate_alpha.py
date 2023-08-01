
import os
import sys
now_path = ["checkpoints/TSL-ETTm2/LightTS/ftM.96_48_96/bs_32/03_25_13_15_34-seed_2021",
            #"checkpoints/TSL-weather/LightTS/TTM.96_48_96/bs_32fwd2_D/03_29_11_36_10-seed_2021",
            #"checkpoints/TSL-weather/DLinear/TTM.96_48_96/bs_32fwd2_D/03_29_11_30_52-seed_2021",
            # "checkpoints/TSL-weather/TimesNet/TTM.96_48_96/bs_32fwd2_D/03_29_11_29_19-seed_2021",
            # "checkpoints/TSL-weather/PatchTST/TTM.96_48_96/bs_32fwd2_D/03_29_11_26_47-seed_2021",
            # "checkpoints/TSL-weather/MICN/TTM.96_96_96/bs_32fwd2_D/03_29_11_20_13-seed_2021",
            # "checkpoints/TSL-weather/Crossformer/TTM.96_48_96/bs_32fwd2_D/03_29_11_18_54-seed_2021",
            # "checkpoints/TSL-weather/LightTS/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_11_15_02-seed_2021",
            # "checkpoints/TSL-weather/DLinear/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_11_09_08-seed_2021",
            # "checkpoints/TSL-weather/TimesNet/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_11_07_59-seed_2021",
            # "checkpoints/TSL-weather/PatchTST/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_11_05_29-seed_2021",
            # "checkpoints/TSL-weather/MICN/TTM.96_96_96/bs_32fwd2_D_Rog5/03_29_10_54_27-seed_2021",
            # "checkpoints/TSL-weather/Crossformer/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_10_54_17-seed_2021",
            # "checkpoints/TSL-weather/MICN/ftM.96_48_96/bs_32/03_25_16_58_06-seed_2021",
            # "checkpoints/TSL-weather/LightTS/ftM.96_48_96/bs_32/03_25_14_45_21-seed_2021",
            # "checkpoints/TSL-weather/DLinear/ftM.96_48_96/bs_32/03_25_14_05_42-seed_2021",
            # "checkpoints/TSL-weather/TimesNet/ftM.96_48_96/bs_32/03_24_11_29_25-seed_2021",
            # "checkpoints/TSL-weather/PatchTST/ftM.96_48_96/bs_32/03_24_10_50_33-seed_2021",
            # "checkpoints/TSL-weather/Crossformer/ftM.96_48_96/bs_32/03_24_10_03_41-seed_2021",
            # "checkpoints/TSL-traffic/LightTS/TTM.96_48_96/bs_32fwd2_D/03_30_10_13_27-seed_2021",
            # "checkpoints/TSL-traffic/DLinear/TTM.96_48_96/bs_32fwd2_D/03_30_09_23_20-seed_2021",
            # "checkpoints/TSL-traffic/PatchTST/TTM.96_48_96/bs_4fwd2_D/03_30_05_26_12-seed_2021",
            # "checkpoints/TSL-traffic/LightTS/TTM.96_48_96/bs_32fwd2_D_Rog5/03_30_05_08_37-seed_2021",
            # "checkpoints/TSL-traffic/Crossformer/TTM.96_96_96/bs_4fwd2_D/03_30_04_36_48-seed_2021",
            # "checkpoints/TSL-traffic/DLinear/TTM.96_48_96/bs_32fwd2_D_Rog5/03_30_04_06_09-seed_2021",
            # "checkpoints/TSL-traffic/PatchTST/TTM.96_48_96/bs_4fwd2_D_Rog5/03_30_02_23_18-seed_2021",
            # "checkpoints/TSL-traffic/TimesNet/TTM.96_48_96/bs_32fwd2_D/03_30_02_13_39-seed_2021",
            # "checkpoints/TSL-traffic/MICN/TTM.96_96_96/bs_32fwd2_D/03_30_01_46_19-seed_2021",
            # "checkpoints/TSL-traffic/Crossformer/TTM.96_96_96/bs_4fwd2_D_Rog5/03_30_00_55_55-seed_2021",
            # "checkpoints/TSL-traffic/TimesNet/TTM.96_48_96/bs_32fwd2_D_Rog5/03_30_00_53_32-seed_2021",
            # "checkpoints/TSL-traffic/MICN/TTM.96_96_96/bs_32fwd2_D_Rog5/03_30_00_44_38-seed_2021",
            # "checkpoints/TSL-traffic/Crossformer/ftM.96_48_96/bs_16/03_26_00_14_12-seed_2021",
            # "checkpoints/TSL-traffic/MICN/ftM.96_48_96/bs_32/03_25_16_54_54-seed_2021",
            # "checkpoints/TSL-traffic/LightTS/ftM.96_48_96/bs_32/03_25_13_59_16-seed_2021",
            # "checkpoints/TSL-traffic/DLinear/ftM.96_48_96/bs_32/03_25_13_30_16-seed_2021",
            # "checkpoints/TSL-traffic/PatchTST/ftM.96_48_96/bs_4/03_25_01_17_26-seed_2021",
            # "checkpoints/TSL-traffic/TimesNet/ftM.96_48_96/bs_32/03_24_06_36_51-seed_2021",
            # "checkpoints/TSL-Exchange/LightTS/TTM.96_48_96/bs_32fwd2_D/03_30_01_37_12-seed_2021",
            # "checkpoints/TSL-Exchange/DLinear/TTM.96_48_96/bs_32fwd2_D/03_30_01_24_24-seed_2021",
            # "checkpoints/TSL-Exchange/TimesNet/TTM.96_48_96/bs_32fwd2_D/03_30_01_21_25-seed_2021",
            # "checkpoints/TSL-Exchange/PatchTST/TTM.96_48_96/bs_32fwd2_D/03_30_01_18_52-seed_2021",
            # "checkpoints/TSL-Exchange/MICN/TTM.96_96_96/bs_32fwd2_D/03_30_01_17_11-seed_2021",
            # "checkpoints/TSL-Exchange/Crossformer/TTM.96_96_96/bs_32fwd2_D/03_30_01_15_40-seed_2021",
            # "checkpoints/TSL-Exchange/LightTS/TTM.96_48_96/bs_32fwd2_D_Rog5/03_30_00_43_50-seed_2021",
            # "checkpoints/TSL-Exchange/DLinear/TTM.96_48_96/bs_32fwd2_D_Rog5/03_30_00_41_25-seed_2021",
            # "checkpoints/TSL-Exchange/TimesNet/TTM.96_48_96/bs_32fwd2_D_Rog5/03_30_00_39_23-seed_2021",
            # "checkpoints/TSL-Exchange/PatchTST/TTM.96_48_96/bs_32fwd2_D_Rog5/03_30_00_38_41-seed_2021",
            # "checkpoints/TSL-Exchange/MICN/TTM.96_96_96/bs_32fwd2_D_Rog5/03_30_00_36_58-seed_2021",
            # "checkpoints/TSL-Exchange/Crossformer/TTM.96_96_96/bs_32fwd2_D_Rog5/03_30_00_36_42-seed_2021",
            # "checkpoints/TSL-Exchange/MICN/ftM.96_48_96/bs_32/03_25_16_52_52-seed_2021",
            # "checkpoints/TSL-Exchange/LightTS/ftM.96_48_96/bs_32/03_25_13_28_58-seed_2021",
            # "checkpoints/TSL-Exchange/DLinear/ftM.96_48_96/bs_32/03_25_13_19_04-seed_2021",
            # "checkpoints/TSL-Exchange/TimesNet/ftM.96_48_96/bs_32/03_24_05_40_24-seed_2021",
            # "checkpoints/TSL-Exchange/PatchTST/ftM.96_48_96/bs_32/03_24_05_36_01-seed_2021",
            # "checkpoints/TSL-Exchange/Crossformer/ftM.96_96_96/bs_32/03_24_05_25_36-seed_2021",
            # "checkpoints/TSL-ETTm1/LightTS/TTM.96_48_96/bs_32fwd2_D/03_29_04_50_37-seed_2021",
            # "checkpoints/TSL-ETTm1/DLinear/TTM.96_48_96/bs_32fwd2_D/03_29_04_38_48-seed_2021",
            # "checkpoints/TSL-ETTm1/TimesNet/TTM.96_48_96/bs_32fwd2_D/03_29_04_37_57-seed_2021",
            # "checkpoints/TSL-ETTm1/PatchTST/TTM.96_48_96/bs_32fwd2_D/03_29_04_28_28-seed_2021",
            # "checkpoints/TSL-ETTm1/MICN/TTM.96_96_96/bs_32fwd2_D/03_29_04_23_47-seed_2021",
            # "checkpoints/TSL-ETTm1/Crossformer/TTM.96_48_96/bs_32fwd2_D/03_29_04_23_12-seed_2021",
            # "checkpoints/TSL-ETTm1/LightTS/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_03_30_31-seed_2021",
            # "checkpoints/TSL-ETTm1/DLinear/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_03_12_55-seed_2021",
            # "checkpoints/TSL-ETTm1/TimesNet/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_03_11_37-seed_2021",
            # "checkpoints/TSL-ETTm1/PatchTST/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_03_06_03-seed_2021",
            # "checkpoints/TSL-ETTm1/MICN/TTM.96_96_96/bs_32fwd2_D_Rog5/03_29_02_57_52-seed_2021",
            # "checkpoints/TSL-ETTm1/Crossformer/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_56_44-seed_2021",
            # "checkpoints/TSL-ETTm1/MICN/ftM.96_48_96/bs_32/03_25_16_46_49-seed_2021",
            # "checkpoints/TSL-ETTm1/LightTS/ftM.96_48_96/bs_32/03_25_12_44_36-seed_2021",
            # "checkpoints/TSL-ETTm1/DLinear/ftM.96_48_96/bs_32/03_25_12_02_59-seed_2021",
            # "checkpoints/TSL-ETTm1/TimesNet/ftM.96_48_96/bs_32/03_23_20_12_49-seed_2021",
            # "checkpoints/TSL-ETTm1/PatchTST/ftM.96_48_96/bs_32/03_23_19_42_56-seed_2021",
            # "checkpoints/TSL-ETTm1/Crossformer/ftM.96_48_96/bs_32/03_23_18_56_31-seed_2021",
            # "checkpoints/TSL-ETTh2/LightTS/TTM.96_48_96/bs_32fwd2_D/03_29_04_22_19-seed_2021",
            # "checkpoints/TSL-ETTh2/DLinear/TTM.96_48_96/bs_32fwd2_D/03_29_04_20_26-seed_2021",
            # "checkpoints/TSL-ETTh2/TimesNet/TTM.96_48_96/bs_32fwd2_D/03_29_04_15_55-seed_2021",
            # "checkpoints/TSL-ETTh2/PatchTST/TTM.96_48_96/bs_32fwd2_D/03_29_04_15_14-seed_2021",
            # "checkpoints/TSL-ETTh2/MICN/TTM.96_96_96/bs_32fwd2_D/03_29_04_13_11-seed_2021",
            # "checkpoints/TSL-ETTh2/Crossformer/TTM.96_48_96/bs_32fwd2_D/03_29_04_12_13-seed_2021",
            # "checkpoints/TSL-ETTh2/LightTS/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_55_38-seed_2021",
            # "checkpoints/TSL-ETTh2/DLinear/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_53_04-seed_2021",
            # "checkpoints/TSL-ETTh2/TimesNet/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_51_37-seed_2021",
            # "checkpoints/TSL-ETTh2/PatchTST/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_49_33-seed_2021",
            # "checkpoints/TSL-ETTh2/MICN/TTM.96_96_96/bs_32fwd2_D_Rog5/03_29_02_47_20-seed_2021",
            # "checkpoints/TSL-ETTh2/Crossformer/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_45_55-seed_2021",
            # "checkpoints/TSL-ETTh2/DLinear/ftM.96_48_96/bs_32/03_25_21_53_41-seed_2021",
            # "checkpoints/TSL-ETTh2/MICN/ftM.96_48_96/bs_32/03_25_16_45_36-seed_2021",
            # "checkpoints/TSL-ETTh2/LightTS/ftM.96_48_96/bs_32/03_25_12_01_15-seed_2021",
            # "checkpoints/TSL-ETTh2/TimesNet/ftM.96_48_96/bs_32/03_23_20_08_28-seed_2021",
            # "checkpoints/TSL-ETTh2/PatchTST/ftM.96_48_96/bs_32/03_23_19_41_04-seed_2021",
            # "checkpoints/TSL-ETTh2/Crossformer/ftM.96_48_96/bs_32/03_23_18_53_24-seed_2021",
            # "checkpoints/TSL-ETTh1/PatchTST/TTM.96_48_96/bs_32fwd2_D_Rog5/04_06_18_43_46-seed_2021",
            # "checkpoints/TSL-ETTh1/TimesNet/TTM.96_48_96/bs_32fwd2_D/03_29_04_08_57-seed_2021",
            # "checkpoints/TSL-ETTh1/PatchTST/TTM.96_48_96/bs_32fwd2_D/03_29_04_06_43-seed_2021",
            # "checkpoints/TSL-ETTh1/MICN/TTM.96_96_96/bs_32fwd2_D/03_29_04_06_04-seed_2021",
            # "checkpoints/TSL-ETTh1/LightTS/TTM.96_48_96/bs_32fwd2_D/03_29_04_05_48-seed_2021",
            # "checkpoints/TSL-ETTh1/DLinear/TTM.96_48_96/bs_32fwd2_D/03_29_04_03_18-seed_2021",
            # "checkpoints/TSL-ETTh1/Crossformer/TTM.96_48_96/bs_32fwd2_D/03_29_04_02_19-seed_2021",
            # "checkpoints/TSL-ETTh1/TimesNet/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_41_42-seed_2021",
            # "checkpoints/TSL-ETTh1/MICN/TTM.96_96_96/bs_32fwd2_D_Rog5/03_29_02_34_56-seed_2021",
            # "checkpoints/TSL-ETTh1/LightTS/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_34_19-seed_2021",
            # "checkpoints/TSL-ETTh1/DLinear/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_30_31-seed_2021",
            # "checkpoints/TSL-ETTh1/Crossformer/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_28_14-seed_2021",
            # "checkpoints/TSL-ETTh1/TimesNet/ftM.96_48_96/bs_32/03_23_20_06_17-seed_2021",
            # "checkpoints/TSL-ETTh1/PatchTST/ftM.96_48_96/bs_32/03_23_19_38_34-seed_2021",
            # "checkpoints/TSL-ETTh1/MICN/ftM.96_96_96/bs_32/03_23_19_22_42-seed_2021",
            # "checkpoints/TSL-ETTh1/LightTS/ftM.96_48_96/bs_32/03_23_19_20_23-seed_2021",
            # "checkpoints/TSL-ETTh1/DLinear/ftM.96_48_96/bs_32/03_23_19_09_00-seed_2021",
            # "checkpoints/TSL-ETTh1/Crossformer/ftM.96_48_96/bs_32/03_23_18_51_17-seed_2021",
            # "checkpoints/TSL-ETTm2/DLinear/ftM.96_48_96/bs_32/03_25_12_49_32-seed_2021",
            # "checkpoints/TSL-ETTm2/TimesNet/ftM.96_48_96/bs_32/03_23_20_17_04-seed_2021",
            # "checkpoints/TSL-ETTm2/PatchTST/ftM.96_48_96/bs_32/03_23_19_44_51-seed_2021",
            # "checkpoints/TSL-ETTm2/MICN/ftM.96_48_96/bs_32/03_25_16_50_34-seed_2021",
            # "checkpoints/TSL-ETTm2/Crossformer/ftM.96_48_96/bs_32/03_23_19_01_42-seed_2021",
            # "checkpoints/TSL-ETTm2/LightTS/TTM.96_48_96/bs_32fwd2_D/03_29_04_00_36-seed_2021",
            # "checkpoints/TSL-ETTm2/DLinear/TTM.96_48_96/bs_32fwd2_D/03_29_03_50_44-seed_2021",
            # "checkpoints/TSL-ETTm2/TimesNet/TTM.96_48_96/bs_32fwd2_D/03_29_03_46_11-seed_2021",
            # "checkpoints/TSL-ETTm2/PatchTST/TTM.96_48_96/bs_32fwd2_D/03_29_03_44_01-seed_2021",
            # "checkpoints/TSL-ETTm2/MICN/TTM.96_96_96/bs_32fwd2_D/03_29_03_37_12-seed_2021",
            # "checkpoints/TSL-ETTm2/Crossformer/TTM.96_48_96/bs_32fwd2_D/03_29_03_35_18-seed_2021",
            # "checkpoints/TSL-ETTm2/LightTS/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_23_39-seed_2021",
            # "checkpoints/TSL-ETTm2/DLinear/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_15_19-seed_2021",
            # "checkpoints/TSL-ETTm2/TimesNet/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_06_20-seed_2021",
            # "checkpoints/TSL-ETTm2/PatchTST/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_02_03_45-seed_2021",
            # "checkpoints/TSL-ETTm2/MICN/TTM.96_96_96/bs_32fwd2_D_Rog5/03_29_01_58_11-seed_2021",
            # "checkpoints/TSL-ETTm2/Crossformer/TTM.96_48_96/bs_32fwd2_D_Rog5/03_29_01_21_45-seed_2021"
            ]
import numpy as np
print(f"we detect {len(now_path)} trail path; \n  from {now_path[0]} to \n  {now_path[-1]}")
from run import get_args
from train import main_worker,check_exist_via_lock,main,get_file_path
for path in now_path:
    if "checkpoint.pth" not in os.listdir(path):
        print(f"no weight in {path} !!!")
        #os.system(f"rm -r {path}")
        continue
    args                   = get_args(os.path.join(path,"config.json"))
    args.mode              = "monitor_alpha"
    args.pretrain_weight   = os.path.join(path,"checkpoint.pth")
    args.fourcast_step     = 22
    args.valid_batch_size  = 64
    args.fourcast_step     = 13 if 'exchange' in args.data_path else 22
    args.use_wandb         = 'wandb_runtime'
    # lock_dir = "/nvme/zhangtianning/share/lock"
    # lock_file = os.path.join(lock_dir, "TSL/alpha",args.SAVE_PATH.replace("/", "-"))+'.lock'
    # if os.path.exists(lock_file):
    #     print(f"====> detected experiment {lock_file} is running in other machine,  ====> skip..............")
    #     continue
    # else:
    #     print(f"====> do not find lock file at {lock_file} , ===> continue..............")
    #     os.system(f"touch {lock_file}")
    main(args)

