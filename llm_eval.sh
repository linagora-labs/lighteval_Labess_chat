lighteval accelerate \
    "pretrained=linagora/Labess-7b-chat-16bit" \
    "examples/tasks/OALL_tasks_tunisian.txt" \
    --custom-tasks "community_tasks/tun_arabic_evals.py" \
    --override-batch-size 1 \
    --output-dir="./evals/"




#usage: lighteval accelerate [-h] (--model_config_path MODEL_CONFIG_PATH | --model_args MODEL_ARGS) [--max_samples MAX_SAMPLES] [--override_batch_size OVERRIDE_BATCH_SIZE]
                            #[--job_id JOB_ID] --output_dir OUTPUT_DIR [--push_results_to_hub] [--save_details] [--push_details_to_hub] [--push_results_to_tensorboard]
                           # [--public_run] [--cache_dir CACHE_DIR] [--results_org RESULTS_ORG] [--use_chat_template] [--system_prompt SYSTEM_PROMPT]
                          #  [--dataset_loading_processes DATASET_LOADING_PROCESSES] [--custom_tasks CUSTOM_TASKS] --tasks TASKS [--num_fewshot_seeds NUM_FEWSHOT_SEEDS]
