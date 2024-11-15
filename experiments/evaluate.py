import argparse
import sys
import math
import random
import json
import shutil
import time
import gc
import os

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from absl import app
from ml_collections import config_flags

from judge_attack import (AttackPrompt,
                        MultiPromptAttack,
                        PromptManager,
                        EvaluateAttack)
from judge_attack import (get_goals_and_targets, get_workers)

_CONFIG = config_flags.DEFINE_config_file('config')


_MODELS = {
    "openchat35": ["openchat/openchat_3.5", {"use_fast": False}, "openchat_3.5", 64],
    "mistral2": ["mistralai/Mistral-7B-Instruct-v0.2",{"use_fast": False}, "mistral", 64],
    "llama2": ["meta-llama/Llama-2-7b-chat-hf",{"use_fast": False}, "llama-2", 64],
    "llama3": ["meta-llama/Meta-Llama-3-8B-Instruct",{"use_fast": False}, "llama-3", 64],
}



def count_true_and_total(lst):
    true_count = sum(sublist.count(True) for sublist in lst)
    total_count = sum(len(sublist) for sublist in lst)
    return true_count, total_count

def count_same_true(lst1, lst2):
    if len(lst1[0]) != len(lst2[0]):
        return "Error: The lists have unequal lengths."
    same_true_count = sum(x and y for x, y in zip(lst1[0], lst2[0]))
    return same_true_count


def main(_):

    params = _CONFIG.value

    with open(params.logfile, 'r') as f:
        log = json.load(f)
    params.logfile = params.logfile.replace('results/', 'eval/')
    controls = log['controls']
    assert len(controls) > 0

    controls = [controls[-1]]   #use control

    train_intro, train_goals1, train_goals2, train_targets, test_intro, test_goals1, test_goals2, test_targets = get_goals_and_targets(params)

    results = {}

    model = params.eval_model

    if model in _MODELS:

        torch.cuda.empty_cache()
        start = time.time()

        params.tokenizer_paths = [
            _MODELS[model][0]
        ]
        params.tokenizer_kwargs = [_MODELS[model][1]]
        params.model_paths = [
            _MODELS[model][0]
        ]
        params.model_kwargs = [
            {"low_cpu_mem_usage": True, "use_cache": True, "temperature": 0.0}
        ]
        params.conversation_templates = [_MODELS[model][2]]
        params.devices = ["cuda:1"]
        batch_size = _MODELS[model][3]

        workers, test_workers = get_workers(params, eval=True)

        managers = {
            "AP": AttackPrompt,
            "PM": PromptManager,
            "MPA": MultiPromptAttack
        }

        attack = EvaluateAttack(
            train_intro,
            train_goals1,
            train_goals2,
            train_targets,
            workers,
            align_weight = params.align_weight,
            enhance_weight = params.enhance_weight,
            perplexity_weight = params.perplexity_weight,
            managers=managers,
            test_intros=test_intro,
            test_goals1=test_goals1,
            test_goals2=test_goals2,
            test_targets=test_targets,
        )

        batch_size = 16
        total_pi, total_em, test_total_jb, test_total_em, total_outputs, test_total_outputs = attack.run(
            range(len(controls)),
            controls,
            batch_size,
            max_new_len=16
        )

        for worker in workers + test_workers:
            worker.stop()

        results[model] = {
            "pi": total_pi,
            "em": total_em,
            "test_jb": test_total_jb,
            "test_em": test_total_em,
            "outputs": total_outputs,
            "test_outputs": test_total_outputs
        }

        true1_count, total1_count = count_true_and_total(total_pi)
        true2_count, total2_count = count_true_and_total(test_total_jb)

        same_true_count = count_same_true(total_pi, test_total_jb)

        prob_1 = true1_count/total1_count
        prob_2 = true2_count/total2_count
        average = (prob_1 + prob_2)/2
        pac = same_true_count/total1_count
        
        results[model]["metric"] ={
            "ASR_half_1": prob_1,
            "ASR_half_2": prob_2,
            "ASR": average,
            "PAC": pac
        } 

        print (f"Model: {model}, ASR: {average}, PAC: {pac}")

        print(f"Saving model results: {model}", "\nTime:", time.time() - start)
        with open(params.logfile, 'w') as f:
            json.dump(results, f)
        
        del workers[0].model, attack
        torch.cuda.empty_cache()


if __name__ == '__main__':
    app.run(main)
