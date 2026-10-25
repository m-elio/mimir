# Mimir

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg.svg)](https://huggingface.co/collections/mimir-lcm/mimir)

Code for the paper "Mimir: Large-Scale Multilingual Concept Modeling" accepted to the main conference of EMNLP 2026.

---

The codebase for training Mimir is mostly based on Meta's codebase and recipes. For simplicity sake, the exact code that was used is provided [here](lcm/). License for the original codebase is provided [here](LCM_LICENSE).

The following are the main changes w.r.t. the original implementation:

* Modified the [dataloader script](lcm/datasets/dataloader.py) to convert sentences into embeddings during training. Additionally, it was also modified for vision embedding extraction, but this part of the code was not used for this project specifically

* Modified the [dataloading script](lcm/datasets/dataloading.py) to create a copy of the sentences column. Since embeddings are extracted during training, a copy of that column is created that is then replaced by the embeddings during training

* Modified the [sentence splitting strategy](lcm/datasets/sentence_splitting.py) to avoid removing characters from some languages (e.g. Japanese)

* Added the normalizer weights obtained from Fineweb-edu and Fineweb 2 [here](lcm/cards/normalizer/normalizer_FINEWEB.pt)

* Modified the [datacards](lcm/datacards/datacards.yaml) to point to pre-training and instruction-tuning data. **NOTE**: the path in these datacards needs to be updated to the local path where your copies of the data are stored. Modify the datacards yaml file by replacing the path to do so

* Some minor stuff (e.g., [config](lcm/datasets/configs.py) was modified to force a local path as data source)

The remaining code is pretty much the same as the original implementation from Meta. Requirements for training are reported [here](lcm_training_requirements.txt).

To use the codebase, it is necessary to split the text of each instance into sentences. Pre-training data split into sentences is provided [here for English](https://huggingface.co/datasets/mimir-lcm/fineweb-edu-350BT-sentence-split) and [here for non-English languages](https://huggingface.co/datasets/mimir-lcm/fineweb-2-sentence-split). Multiple scripts were created to handle different datasets that had different columns and features. An example of script used for sentence splitting is provided [here](scripts/prepare_dataset.py).

The training recipes that were used can be found [here](recipes).

To launch training, an example bash script is provided [here for pre-training](train.sh) and [here for instruction tuning](train_it.sh). In this case, the Leonardo Supercomputer was used and the script was launched as a job through slurm. For Leonardo specifically, the [load](load.sh) file containing all modules that need to be loaded to launch training on that specific supercomputer is also provided.

After completing pre-training, add the model card obtained from the corresponding checkpoint [here](lcm/cards/) and load the pre-trained model using the corresponding name from the model card when launching the instruction-tuning script.

Finally, it is highly recommended to download all needed models before training/inference. Specifically, you should download the [sat-3l](https://huggingface.co/segment-any-text/sat-3l) model for wtp-split (sentence splitting library) and the [SONAR encoder and decoder](https://github.com/facebookresearch/SONAR) models through the corresponding pipeline.

---

For evaluation, the scripts used for inference are provided [here](inference_pretraining.py) for pre-training and [here](inference_it.py) for instruction tuning. Requirements for evaluation are reported [here](eval_requirements.txt).

The pre-training inference script also calculates the L2, R-L2, CA and PAR metrics, while the instruction-tuning inference script only generates the text.

Instruction-tuning outputs are evaluated using the ROUGE-L implementation provided [here](evaluate_rougel.py). The script leverages a mBERT tokenizer to support multilingual inputs, similarly to what has been done in [this](https://github.com/faisaltareque/Multilingual-Rouge-Scorer) ROUGE-L implementation.

Aggregated results per language are provided [here](results/) for both pre-training and instruction tuning.

---

If you use or find useful any part of this codebase, please cite the following

```bibtex
@inproceedings{musacchio-etal-2026-mimir,
    title = "Mimir: Large-Scale Multilingual Concept Modeling",
    author = "Musacchio, Elio  and
      Siciliani, Lucia  and
      Basile, Pierpaolo",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.569/",
    pages = "13051--13073",
    ISBN = "979-8-89176-459-0",
    abstract = "Current language modeling approaches are built around tokens. Text corpora are split into tokens, and models are trained by performing computations on these tokens, such as predicting the next token given the preceding ones as context. This paradigm has become the standard in modern language modeling, especially given the outstanding performance obtained by token-based architectures. However, recent works have not only begun to question how language models process and understand meaning from tokens, but also to question whether using higher levels of granularity could advance the research field. This led to the idea of Concept Modeling, that is, to directly train models for next-concept prediction rather than next-token prediction. In this work, we introduce Mimir, a 1.6B Large Concept Model trained for multilingual concept understanding and generation. We leverage a large-scale multilingual pre-training corpus (38,883,987,240 sentences) spanning 46 languages and a large-scale multi-turn and multilingual instruction-tuning dataset (66,816,428 sentences) covering a total of 35 languages. We extensively evaluate model performance against a language model with a comparable number of parameters."
}
```
