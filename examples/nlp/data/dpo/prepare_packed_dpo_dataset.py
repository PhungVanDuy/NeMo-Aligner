# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import torch

from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING, Tuple
from tqdm import tqdm

import numpy as np

from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.sequence_packing_utils import create_hist, create_packing_strategy

from nemo_aligner.data.nlp.builders import build_train_valid_test_dpo_datasets
from nemo_aligner.data.nlp.datasets import DPOModelDataset

if TYPE_CHECKING:
    from omegaconf import DictConfig

""" 
TODO: DOCSTRING
"""


def tokenize_dataset(cfg: 'DictConfig'):
    """
    TODO: DOCS
    """

    logging.info("Tokenizing dataset...")
    # using the same template as SFT/PEFT script. This may be overkill but guarantees the preprocess settings
    # are identical to normal SFT training

    ## TODO: fix this! hf tokenizer path doesn't need to be a dir
    #if os.path.isdir(cfg.tokenizer_path):
    # pass in a Hugging Face folder which contains tokenizer.json
    tokenizer = get_nmt_tokenizer(library="huggingface", model_name=cfg.tokenizer_path, use_fast=True)
    #else:
    #    tokenizer = get_nmt_tokenizer(library="sentencepiece", tokenizer_model=cfg.tokenizer_path)

    with open(cfg.model.data.data_prefix, "r", encoding="utf_8") as fr:
        data_payload = [json.loads(line.strip()) for line in fr]
    documents = np.arange(len(data_payload), step=1, dtype=np.int32)
    dataset = DPOModelDataset(
        cfg=cfg.model,
        name="packing_dataset",
        tokenizer=tokenizer,
        data_prefix=cfg.model.data.data_prefix, ## TODO: fix
        documents=documents,
        data=data_payload,
        seq_length=cfg.model.data.seq_length,
        seed=1234,
        drop_last=True, ## TODO: do not hard-code
    )

    combined_dataset = []
    for item in dataset:
        for k in item:
            if isinstance(item[k], torch.Tensor):
                item[k] = item[k].numpy()
        item["input_ids"] = item["chosen"] ## WAR for create_hist
        combined_dataset.append(item)

    ## NOTE: chosen and rejected are already padded to the same sequence length,
    ## so there's nothing to do here!
    return np.array(combined_dataset)


## modified version of https://github.com/NVIDIA/NeMo/blob/main/nemo/utils/sequence_packing_utils.py#L178 for DPO
## pack size should be at least 2*max_seq_length since the packed sequences include both the chosen and rejected sequences
## for a given example
def fill_packing_strategy(
    assignments: List[List[int]], sequences: Dict[int, List[Dict]], pack_size: int
) -> List[Dict]:
    """
    Fills the packing strategy with actual sequence data based on assignments and sequence information.

    This function takes the assignments generated by the packing algorithm (containing sequence length indices),
    the original sequences data, and the pack size. It iterates through the assignments, retrieves the corresponding
    sequences from the sequences dictionary, and constructs the final output data structure with input IDs, loss masks
    (if available), and starting indices for each sequence in a packed sequence.

    Args:
          assignments: A list of lists, where each inner list represents a bin and contains the indices of the
                        sequence lengths assigned to that bin (output of 'create_packing_strategy').
          sequences: A dictionary where keys are sequence lengths and values are lists of corresponding sequences
                      from the dataset (output of 'create_hist').
          pack_size: The maximum capacity of each bin.

    Returns:
          output_data: A list of dictionaries, where each dictionary represents a packed sequence with its input IDs,
                        loss mask (if available), and starting indices.
    """
    ifile_handles = dict()
    for seq_len in tqdm(range(pack_size + 1)):
        per_seq_data = sequences[seq_len]
        if len(per_seq_data) > 0:
            perm = np.random.permutation(len(per_seq_data))

            chosen_tokens = np.array([x['chosen'] for x in per_seq_data])[perm].tolist()
            rejected_tokens = np.array([x['rejected'] for x in per_seq_data])[perm].tolist()
            chosen_labels = np.array([x['chosen_labels'] for x in per_seq_data])[perm].tolist()
            rejected_labels = np.array([x['rejected_labels'] for x in per_seq_data])[perm].tolist()
            chosen_length = np.array([x['chosen_length'] for x in per_seq_data])[perm].tolist()
            rejected_length = np.array([x['rejected_length'] for x in per_seq_data])[perm].tolist()
            chosen_reward = np.array([x['chosen_reward'] for x in per_seq_data])[perm].tolist()
            rejected_reward = np.array([x['rejected_reward'] for x in per_seq_data])[perm].tolist()

            ifile_handles[seq_len] = (
                chosen_tokens,
                rejected_tokens,
                chosen_labels,
                rejected_labels,
                chosen_length,
                rejected_length,
                chosen_reward,
                rejected_reward,
            )

    (
        chosen_ids,
        rejected_ids,
        chosen_labels,
        rejected_labels,
        chosen_length,
        rejected_length,
        chosen_reward,
        rejected_reward,
        seq_boundaries,
    ) = {}, {}, {}, {}, {}, {}, {}, {}, {}

    for oindex, assignment in tqdm(enumerate(assignments), total=len(assignments)):
        (
            _chosen_ids,
            _rejected_ids,
            _chosen_labels,
            _rejected_labels,
            _chosen_length,
            _rejected_length,
            _chosen_reward,
            _rejected_reward,
            _seq_boundaries
         ) = [], [], [], [], [], [], [], [], [0]

        for seq_length in assignment:

            previous_seq_len = len(_chosen_ids) ## also equals len(_rejected_ids)

            _chosen_ids.extend(ifile_handles[seq_length][0].pop())
            _rejected_ids.extend(ifile_handles[seq_length][1].pop())
            _chosen_labels.extend(ifile_handles[seq_length][2].pop())
            _rejected_labels.extend(ifile_handles[seq_length][3].pop())
            _chosen_length.append(ifile_handles[seq_length][4].pop())
            _rejected_length.append(ifile_handles[seq_length][5].pop())
            _chosen_reward.append(ifile_handles[seq_length][6].pop())
            _rejected_reward.append(ifile_handles[seq_length][7].pop())

            ## store the boundaries for the chosen, rejected sequences
            _seq_boundaries.append(len(_chosen_ids))

        chosen_ids[oindex] = _chosen_ids
        rejected_ids[oindex] = _rejected_ids
        chosen_labels[oindex] = _chosen_labels
        rejected_labels[oindex] = _rejected_labels
        chosen_length[oindex] = _chosen_length
        rejected_length[oindex] = _rejected_length
        chosen_reward[oindex] = _chosen_reward
        rejected_reward[oindex] = _rejected_reward
        seq_boundaries[oindex] = _seq_boundaries #[:-1]

    output_data = []
    for i in range(len(chosen_ids)):
        item_dict = {
            'chosen': chosen_ids[i],
            'rejected': rejected_ids[i],
            'chosen_labels': chosen_labels[i],
            'rejected_labels': rejected_labels[i],
            'chosen_length': chosen_length[i],
            'rejected_length': rejected_length[i],
            'chosen_reward': chosen_reward[i],
            'rejected_reward': rejected_reward[i],
            'seq_boundaries': seq_boundaries[i]
        }
        print(f'{item_dict["seq_boundaries"]=}')
        output_data.append(item_dict)

    for i in range(8):
        assert all(not seq[i] for seq in ifile_handles.values()), f"Error: There are items left over from the assignment. {ifile_handles.values()=}"
    return output_data

@dataclass
class PackingArgs:
    output_dir: str = "output"
    pack_sizes: Tuple[int] = (2048,)
    packing_algorithm: str = "first_fit_shuffle"
    seed: int = 0

    def from_config(self, cfg: 'DictConfig'):
        for required_arg in ('output_dir', 'pack_sizes'):
            assert cfg.get(required_arg, None), f"Please specify +{required_arg}=..."
        self.output_dir = cfg.output_dir
        self.pack_sizes = cfg.pack_sizes
        self.packing_algorithm = cfg.get("packing_algorithm", "first_fit_shuffle")
        self.seed = cfg.get("seed", 0)
        return self


@hydra_runner(
    config_path="../../gpt/conf", config_name="gpt_dpo"
)
def main(cfg: 'DictConfig') -> None:
    args = PackingArgs().from_config(cfg)
    dataset = tokenize_dataset(cfg)
    sequences, histogram = create_hist(dataset, 2*cfg.model.data.seq_length) ## multiply by 2 because packed sequences include chosen and rejected
    for pack_size in args.pack_sizes:
        assignments = create_packing_strategy(histogram, pack_size, args.packing_algorithm)
        output_data = fill_packing_strategy(assignments, sequences, pack_size)

        # save output data
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f'packed_{pack_size}_seed{args.seed}.npy')
        np.save(output_path, output_data)
        logging.info(f"Done, output written to {output_path}")

### TODO: Update!!
    logging.info(
        f"""
✅ Packed datasets with pack sizes {args.pack_sizes} are prepared successfully. 
To train with packed sequences, you need to make changes to the DPO config file. See NeMo Documentation 
for more details: <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/throughput_optimizations.html#sequence-packing-for-sft-peft>
"""
    )


if __name__ == '__main__':
    main()