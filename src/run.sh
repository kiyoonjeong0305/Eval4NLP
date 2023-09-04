model_list="orca_mini_v3_7b OpenOrca-Platypus2-13B Platypus2-70B-Instruct-GPTQ"

for model in ${model_list}
do
    python inference.py \
    --exp_name prompt_mk3 \
    --model_name ${model} \
    --prompt_path /Workspace/ky/SharedTask2023/prompt/dynamic_summary_mk2 \
    --device cuda:1 \
    --data_type train 
done

