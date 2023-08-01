# for script_name in `ls $1`
# do
#     echo "bash $1/$script_name"
#     #bash $1/$script_name
# done

bash scripts/long_term_forecast/custom/DLinear/DLinear_ETTh2.sh
bash scripts/long_term_forecast/custom/DLinear/DLinear_ETTm1.sh
bash scripts/long_term_forecast/custom/DLinear/DLinear_ETTm2.sh
bash scripts/long_term_forecast/custom/DLinear/DLinear_Exchange.sh
bash scripts/long_term_forecast/custom/DLinear/DLinear_Traffic.sh
bash scripts/long_term_forecast/custom/DLinear/DLinear_Weather.sh