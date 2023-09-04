# model_list="Nous-Hermes-13b guanaco-65B-GPTQ WizardLM-13B-V1.1-GPTQ"
model=WizardLM-13B-V1.1-GPTQ
task=zh_en
outputfile_name=dev_cot_sample.tsv

# CUDA_VISIBLE_DEVICES=1 python src/inference.py \
# --model_name ${model} \
# --model_path /code/SharedTask2023/models/${model}

python src/inference.py \
--model_name ${model} \
--model_path /code/SharedTask2023/models/${model} \
--data_path /code/SharedTask2023/data/${task}/dev_${task}.tsv \
--output_path /code/SharedTask2023/data/${task}/result/${outputfile_name} \
--task ${task} \
--target_lang English \
--source_lang Chinese \

