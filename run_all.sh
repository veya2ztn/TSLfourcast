for GPU in 0 1 2 3 4 5 6 7
do 
#export CUDA_VISIBLE_DEVICES=$GPU;nohup bash batch_run.sh > log/benchmark.$CUDA_VISIBLE_DEVICES.1.log &
export CUDA_VISIBLE_DEVICES=$GPU;nohup wandb agent ai4quake/TSL-ETTm2/oxk9k36z --count 5 > log/benchmark.$CUDA_VISIBLE_DEVICES.1.log &


sleep 5
done
