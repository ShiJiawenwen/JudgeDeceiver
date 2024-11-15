from ml_collections import config_dict

def get_config():
    config = config_dict.ConfigDict()

    # General parameters 
    config.target_weight=1.0
    config.align_weight = 1.0
    config.enhance_weight=1.0
    config.perplexity_weight=0.1
    config.control_weight=0.1
    config.progressive_goals=False 
    config.progressive_models=False
    config.anneal=False
    config.incr_control=False
    config.stop_on_success=False
    config.verbose=True
    config.allow_non_ascii=False
    config.num_train_models=1

    # Results
    config.result_prefix = '../results/test'

    # tokenizers
    config.tokenizer_paths=['meta-llama/Meta-Llama-3-8B-Instruct']
    config.tokenizer_kwargs=[{"use_fast": False}]
    
    config.model_paths=['meta-llama/Meta-Llama-3-8B-Instruct']
    config.model_kwargs=[{"low_cpu_mem_usage": True, "use_cache": False}]
    config.conversation_templates=['llama-3']
    config.devices=['cuda:0']

    # data
    config.train_data = ''
    config.test_data = ''
    config.n_train_data = 6
    config.n_test_data = 6
    config.data_offset = 0

    # attack-related parameters
    config.attack = 'gcg'
    config.control_init = "correct correct correct correct correct correct correct correct correct correct correct correct correct correct correct correct correct correct correct correct"
    config.n_steps = 10
    config.test_steps = 1
    config.batch_size = 256
    config.lr = 0.01
    config.topk = 256
    config.temp = 1
    config.filter_cand = True

    config.gbda_deterministic = True

    #eval parameters
    config.eval_model = ''

    return config
