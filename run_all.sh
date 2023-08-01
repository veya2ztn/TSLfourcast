for GPU in 0 1 2 3 4 5 6 7 
do 
export CUDA_VISIBLE_DEVICES=$GPU;nohup bash test.sh > log/benchmark.$CUDA_VISIBLE_DEVICES.1.log &
sleep 5
done
