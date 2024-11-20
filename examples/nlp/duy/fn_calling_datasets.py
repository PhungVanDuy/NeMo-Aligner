from datasets import load_dataset

lst_names = [
    "slm-research-vn/hermes-fn-call-v3",
    "slm-research-vn/tiny-agent-fn-call-v3-no-response",
    "slm-research-vn/glaive-fn-call-v3",
    "slm-research-vn/xlam-fn-call-v3",
    "slm-research-vn/tiny-agent-fn-call-v3-synthetic-augmented",
    "slm-research-vn/tiny-agent_multi-turn_Sonnet",
    "slm-research-vn/magpie-ultra-v0.1_with_system_prompt",
    "slm-research-vn/HelpSteer2_with_system_prompt"
]

lst_ds = []

for name in lst_names:
    dataset = load_dataset(name)["train"]
    dataset = dataset.select_columns(["conversations"])
    lst_ds.append(dataset)

from datasets import concatenate_datasets

total_ds = concatenate_datasets(lst_ds)
import ipdb; ipdb.set_trace()