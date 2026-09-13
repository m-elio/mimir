import os
import torch
import shutil
import datasets
import argparse

from datasets import load_dataset
from multiprocess import set_start_method
from st_split import (
    SentenceSplitterConfig,
    SentenceSplitter
)

bs = 1024

splitter_config = SentenceSplitterConfig(
    columns=["text"],
    model_name="sat-3l",
    verbose=True,
    sentence_threshold=0.02,
    max_sentence_len=256,
    batch_size=bs
)
text_splitter = SentenceSplitter(splitter_config)

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sub_string')
parser.add_argument('-l', '--lang_code')
args = parser.parse_args()

substring = args.sub_string
lang_code = args.lang_code

print(substring)
print(lang_code)


def emb_extr(x, rank):

    device = f"cuda:{rank}"

    text_splitter.model.to(device)

    split_text_original = text_splitter(x)["text_sentences"].to_pylist()
    print(split_text_original[0])
    print(lang_code)

    return {'id': list(x['id']), 'text': list(x['text']), 'text_sentences': split_text_original, 'lang': [lang_code for _ in range(len(x['id']))]}


if __name__ == "__main__":

    set_start_method('spawn')
    datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

    for x in os.listdir("edu-350BT/sample/350BT"):

        if ".parquet" in x and substring in x:

            if os.path.isfile(os.path.join("edu-350BT-split", x)):
                continue

            ds_original = load_dataset("parquet", data_files={"train": os.path.join("edu-350BT/sample/350BT", x)}, cache_dir=f"home/fineweb_cache_{x}", split="train", num_proc=8)
            ds = ds_original.with_format("arrow")

            ds = ds.map(emb_extr, with_rank=True, num_proc=torch.cuda.device_count(), batched=True, batch_size=bs)
            ds.to_parquet(os.path.join("edu-350BT-split", x))
            shutil.rmtree(f"home/fineweb_cache_{x}")