import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse

import lib.perturbations as perturbations
import lib.defenses as defenses
import lib.attacks as attacks
import lib.language_models as language_models
import lib.model_configs as model_configs

import json


def main(args):
    with open(args.results_dir, 'w') as f:
        pass

    # Instantiate the targeted LLM
    config = model_configs.MODELS[args.target_model]
    target_model = language_models.LLM(
        model_path=config['model_path'],
        tokenizer_path=config['tokenizer_path'],
        conv_template_name=config['conversation_template'],
        device='cuda:0'
    )

    # Create Corrupt instance
    defense = defenses.CCSmoothing(
        target_model=target_model,
        pert_type=args.corrupt_pert_type,
        pert_pct=args.corrupt_pert_pct,
        num_copies=args.corrupt_num_copies,
        correct=args.correct
    )
    # Create attack instance, used to create prompts
    attack = vars(attacks)[args.attack](
        logfile=args.attack_logfile,
        target_model=target_model
    )

    jailbroken_results = []
    jailbroken_results_corrupt = []
    json_results = []

    for i, prompt in tqdm(enumerate(attack.prompts)):

        with open(args.results_dir, "w") as f:
            json.dump(json_results, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {args.results_dir}")

        output, input_item, corrupt_output, corrupt_input, jb, jb_corrupt = defense(prompt)

        jailbroken_results.append(jb)

        if args.correct and args.corrupt_pert_pct != 0:
            jailbroken_results_corrupt.append(jb_corrupt)
            result = {
                "index": i,
                "corrupt_input": corrupt_input,
                "corrupt_out": corrupt_output,
                "corrupt_correct_input": input_item,
                "corrupt_correct_out": output,
                "corrupt_jailbroken": jb_corrupt,
                "corrupt_correct_jailbroken": jb,
            }
            json_results.append(result)
        else:
            result = {
                "index": i,
                "input": input_item,
                "output": output,
                "is_jailbroken": jb
            }
            json_results.append(result)

    with open(args.results_dir, "w") as f:
        json.dump(json_results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {args.results_dir}")


if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results'
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=0
    )

    # Targeted LLM
    parser.add_argument(
        '--target_model',
        type=str,
        default='vicuna',
        choices=['vicuna', 'llama2', 'gpt3.5']
    )

    # Attacking LLM
    parser.add_argument(
        '--attack',
        type=str,
        default='GCG',
        choices=['GCG', 'PAIR']
    )
    parser.add_argument(
        '--attack_logfile',
        type=str,
        default='data/GCG/vicuna_behaviors.json'
    )

    # Corrupt
    parser.add_argument(
        '--corrupt_num_copies',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--corrupt_pert_pct',
        type=int,
        default=10
    )
    parser.add_argument(
        '--correct',
        action='store_true',
        help='Enable or disable Correct functionality (default: disabled).'
    )
    parser.add_argument(
        '--corrupt_pert_type',
        type=str,
        default='RandomSwapPerturbation',
        choices=[
            'RandomSwapPerturbation',
            'RandomPatchPerturbation',
            'RandomInsertPerturbation',
            'WordSubstitutePerturbation',
            'WordDeletePerturbation',
            'WordInsertPerturbation',
        ]
    )

    args = parser.parse_args()
    main(args)