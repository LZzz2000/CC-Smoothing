# CC-Smoothing

Certified Jailbreaking Defense via Corrupt-and-Correct Smoothing



## Installation

1. Clone this repository and navigate to CC-Smoothing folder

2. Create a new conda environment and install the packages.

```
conda create -n cc-smooth python=3.10
conda activate cc-smooth
pip install -e.
```

3. Download the weights for [Vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) and [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main). Change the ```model_path``` and ```tokenizer_path``` in ```lib/model_configs.py```.
4. Download pretrained models of `neuspell`. You can find these Checkpoints [here](https://drive.google.com/drive/folders/1jgNpYe4TVSF4mMBVtFh4QfB2GovNPdh7). 



## Get API Key

Please get ```api_key``` from [OpenAI API](https://platform.openai.com/docs/overview).

```python
openai.api_key = '' # add your api_key
```



## Experiments

1. You can run CC-Smoothing by running:


```python
python main.py \
    --results_dir ./results \
    --target_model vicuna \
    --attack GCG \
    --attack_logfile ./data/advbench_gcg_vicuna.json \
    --corrupt_pert_type RandomSwapPerturbation \
    --corrupt_pert_pct 5 \
    --corrupt_num_copies 5 \
    --correct
```

2. You can turn the correction function on or off using the switch `--correct`, and control the number of copies and the intensity of perturbations with `corrupt_num_copies` and `corrupt_pert_pct`. You can choose from six perturbation methods: `RandomSwapPerturbation`, `RandomPatchPerturbation`, `RandomInsertPerturbation`, `WordSubstitutePerturbation`, `WordDeletePerturbation`, and `WordInsertPerturbation`. For more details, please refer to `main.py`.

