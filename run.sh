# model_list="Nous-Hermes-13b guanaco-65B-GPTQ WizardLM-13B-V1.1-GPTQ"
# task_list="summarization en_de zh_en"
# data_type_list="train dev test"

model_list="Nous-Hermes-13b guanaco-65B-GPTQ WizardLM-13B-V1.1-GPTQ"
# model=guanaco-65B-GPTQ
data_type=train
promptfile_name=zero_shot_five_ws_sum

if [ $task=summarization ]; then
    prompt_path=/code/SharedTask2023/data/prompt/${task}/${promptfile_name}.txt
else
    prompt_path=/code/SharedTask2023/data/prompt/mt/${promptfile_name}.txt
fi
echo $prompt_path

# python src/inference.py \
# --model_name ${model} \
# --model_path /code/SharedTask2023/models/${model} \
# --data_path /code/SharedTask2023/data/${task}/${data_type}_${task}.tsv \
# --output_path /code/SharedTask2023/results/${task}/${model}/${data_type}/${promptfile_name}.tsv \
# --prompt_path ${prompt_path} \
# --task ${task} \
# --target_lang English \
# --source_lang Chinese \
# --device cuda:7 \
# --data_type ${data_type} 


for model in ${model_list}
do
    python src/inference.py \
    --model_name ${model} \
    --model_path /code/SharedTask2023/models/${model} \
    --data_path /code/SharedTask2023/data/${task}/${data_type}_${task}.tsv \
    --output_path /code/SharedTask2023/results/${task}/${model}/${data_type}/${promptfile_name}.tsv \
    --prompt_path ${prompt_path} \
    --task ${task} \
    --target_lang English \
    --source_lang Chinese \
    --device cuda:1 \
    --data_type ${data_type} 
done

