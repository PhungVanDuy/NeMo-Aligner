# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from collections import defaultdict
from statistics import mean

import torch
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingRandomBatchSampler,
)
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.utils import logging
from nemo_aligner.utils.distributed import SyncTimer
from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.trainer_utils import check_progress, compute_limit_batches
from nemo_aligner.utils.utils import clear_memory
from nemo_aligner.algorithms.dpo import DPOTrainer

def kto_custom_collate(batch, eos_id, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False):
    sample_tokens = [item["sample"] for item in batch]
    sample_lengths = torch.LongTensor([item["sample_length"] for item in batch])
    sample_labels = [item["sample_labels"] for item in batch]
    sample_preference = [item["preference"] for item in batch]

    sample_tokens = torch.nn.utils.rnn.pad_sequence(sample_tokens, batch_first=True, padding_value=eos_id)
    sample_labels = torch.nn.utils.rnn.pad_sequence(sample_labels, batch_first=True, padding_value=-100)

    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        sample_tokens, eos_id, reset_position_ids, reset_attention_mask, eod_mask_loss,
    )
    assert attention_mask.ndim == 4, "attention_mask is incorrect shape for dpo_custom_collate"
    if attention_mask.shape[0] == 1:
        # using .expand() here causes errors from pin_memory=True, so need to use .repeat()
        # attention_mask = attention_mask.expand(len(batch), *((-1,) * (len(attention_mask.shape) - 1)))
        attention_mask = attention_mask.repeat(len(batch), *((1,) * (len(attention_mask.shape) - 1)))

    output = {
        "sample": sample_tokens,
        "sample_length": sample_lengths,
        "sample_labels": sample_labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "preference": sample_preference,
    }
    return output


class KTOTrainer(DPOTrainer):
    """Trainer to coordinate KTO training
    """

    def __init__(
        self,
        cfg: DictConfig,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        logger,
        ckpt_callback,
        run_timer,
    ):
        super().__init__(cfg, model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader, logger, ckpt_callback, run_timer)

    def augment_dataloader(self, dataloader):
        """Augment dataloader with ref policy log prob"""
        iter_dataloader = iter(dataloader)
        buffer = []
        done = False
        while not done:
            try:
                batch = next(iter_dataloader)
            except StopIteration:
                done = True
            else:
                buffer.append(batch)
            if (done and buffer) or len(buffer) == 1:
                logprobs = self.model.get_ref_policy_logprobs(buffer).cpu()
                start = 0
                for batch in buffer:
                    batch_size = len(batch["sample"])
                    batch[f"ref_policy_log_probs"] = logprobs[start : start + batch_size]
                    start += batch_size
                    yield batch
                buffer.clear()
                del logprobs