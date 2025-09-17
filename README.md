# JudgeDeceiver
Official repo of [ACM CCS 2024](https://www.sigsac.org/ccs/CCS2024/home.html) paper: [Optimization-based Prompt Injection Attack to LLM-as-a-Judge](https://arxiv.org/abs/2403.17710).

<div align="center">
<img src="image/main_example.png">
</div>

## News
- 🚀 [Nov 15, 2024]: We release the code of our CCS paper. If you have any questions, feel free to open an issue or email 📧 me. 
- 🎤 [Oct 15, 2024]: We presented our paper at ACM CCS 2024.
- 🎉 [Aug 19, 2024]: Our paper has been accepted to ACM CCS 2024!

## Quick Usage Guide

### 1. Environment Setup
First, ensure you're using Python 3.10 or higher. 

Clone the official repository from [FastChat](https://github.com/lm-sys/FastChat) to install ```fschat```:

```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip3 install --upgrade pip  # Enable PEP 660 support
pip3 install -e ".[model_worker,webui]"
```

Then, install the required dependencies. You can do this easily using pip:
```bash
pip install -e.
```

### 2. Dataset 
We are releasing the datasets used in our experiments, which include **MT-Bench, LLMBar**, and three case studies: **LLM-powered Search, RLAIF (Reinforcement Learning from AI Feedback)**, and **Tool Selection**. Additionally, we are providing all the **optimized suffixes** used in our experiments.

The dataset is organized into the following folder structure:
```bash
dataset/
    ├── data_for_train/           # Training data
    │   ├── basic/                # Training data for main results
    │   │   ├── mtbench/
    │   │   ├── llmbar/
    ├── data_for_eval/            # Test data
    │   ├── basic/                # Test data for main results
    │   │   ├── mtbench/      
    │   │   ├── llmbar/  
    │   ├── case_study/           # Test data for case study
    │   │   ├── search/   
    │   │   ├── rlaif/            
    │   │   ├── metatool/         
    ├── results_suffix/           # Optimized suffix for attack
    │   ├── basic/                # Suffix optimized with 4 models
    │   │   ├── mtbench.json      
    │   │   ├── llmbar.json       
    │   ├── case_study/           # Suffix for case study
    │   │   ├── search.json       
    │   │   ├── rlaif.json        
    │   │   ├── metatool.json     
```

### 3. Optimizing Sequences for Prompt Injection Attacks
To train a suffix for a target question-answer pair to launch a prompt injection attack on LLM-as-a-Judge, follow these steps:

```bash
cd JudgeDeceiver
cd experiments/basic_scripts
```
Run the attack script with the necessary arguments:
```bash
bash run_attack.sh <model_name> <question_id> <device_id> <dataset_name>
```
**Arguments** for run_attack.sh:
- **model_name**: The name of the model to be used (e.g., llama3).
- **question_id**: The ID of the target question-answer pair (e.g., 1).
- **device_id**: The ID of the device to run the attack on (e.g., 0).
- **dataset_name**: The name of the dataset (e.g., llmbar).

For example, the following command:
```bash
bash run_attack.sh llama3 1 0 llmbar
```


### 4. Evaluation
We provide all the suffixes used in our experiments. To evaluate the attack on the **MTBench** and **LLMBar** datasets, follow the steps below:
```bash
cd JudgeDeceiver
cd experiments/basic_scripts
```
Run the evaluation script with the necessary arguments:
```bash
bash run_eval.sh <model_name> <dataset_name>
```
**Examples:**
For instance, to evaluate using the ```llama3``` model on the ```llmbar``` dataset, you would run:
```bash
bash run_attack.sh llama3 llmbar
```

For the three **case study** scenarios, please run the evaluation with additional parameters defined in the Python script. The script requires the following arguments:
- **--scenario:** Specifies the scenario being tested (e.g., search, rlaif, or metatool).
- **--test_id:** The ID of the target question in the dataset.
- **--samples:** The number of candidate answers in the selection pool.

To evaluate the case studies, follow these steps:
```bash
cd JudgeDeceiver
cd experiments/case_study
python evaluate_casestudy.py
```


## Acknowledgement
We sincerely thank [Zou et al. (2023)](https://arxiv.org/abs/2307.15043) for open-sourcing their code!

Our code used [MTBench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) benchmark.

Our code used [LLMBar](https://github.com/princeton-nlp/LLMBar) benchmark.

Our code uses the [MetaTool](https://github.com/HowieHwong/MetaTool/tree/master) benchmark to evaluate attacks on tool selection.

## Citation
If you use this code, please cite the following [paper](https://arxiv.org/abs/2403.17710)

```tex
@inproceedings{shi2024optimization,
  title={Optimization-based prompt injection attack to llm-as-a-judge},
  author={Shi, Jiawen and Yuan, Zenghui and Liu, Yinuo and Huang, Yue and Zhou, Pan and Sun, Lichao and Gong, Neil Zhenqiang},
  booktitle={Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security},
  pages={660--674},
  year={2024}
}
```
