import json
import os
import torch
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

from lcm.models.two_tower_diffusion_lcm.builder import (
    create_two_tower_diffusion_lcm_model,
)
from lcm.models.two_tower_diffusion_lcm.archs import two_tower_diffusion_lcm_1_6B, two_tower_diffusion_lcm_7B
from lcm.inference.two_tower_diffusion_lcm.generator import (
    TwoTowerDiffusionLCMGenerator,
    DiffusionLCMGeneratorOptions,
)
from lcm.datasets.batch import EmbeddingsBatch
from fairseq2.nn.padding import pad_seqs

from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline

import argparse
import torch.nn.functional as F

# There is a bug in the original codebase which does not allow to load the model in torch.float16
# This is because the initial noise vector is always initialized to float32 and never casted to float16

# Import the specific class that contains the bug
from lcm.models.two_tower_diffusion_lcm.builder import TwoTowerDiffusionLCModel

# 1. Save the original, buggy method
_original_sample_fn = TwoTowerDiffusionLCModel.sample_initial_noise_vectors

# 2. Define a wrapper that casts the output to the correct precision
def _patched_sample_fn(self, batch_size: int):
    # Call the original method to get the (accidentally float32) latents
    latents = _original_sample_fn(self, batch_size)
    
    # Intercept and force them to match the model's actual precision
    return latents.to(dtype=self.dtype)

# 3. Permanently overwrite the method in the class
TwoTowerDiffusionLCModel.sample_initial_noise_vectors = _patched_sample_fn

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint_path', required=True)
parser.add_argument('-d', '--ds', required=True)
args = parser.parse_args()

LANGS = [
    "arb_Arab",
    "bul_Cyrl",
    "cat_Latn",
    "ces_Latn",
    "dan_Latn",
    "deu_Latn",
    "eng_Latn",
    "fra_Latn",
    "hin_Deva",
    "ind_Latn",
    "ita_Latn",
    "jpn_Jpan",
    "kor_Hang",
    "nld_Latn",
    "por_Latn",
    "ron_Latn",
    "rus_Cyrl",
    "slk_Latn",
    "slv_Latn",
    "spa_Latn",
    "srp_Cyrl",
    "swe_Latn",
    "tha_Thai",
    "tur_Latn",
    "ukr_Cyrl",
    "vie_Latn",
    "zho_Hans"
]
    
DS = args.ds
CHECKPOINT_PATH = args.checkpoint_path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_BATCH_SIZE = 250

TEXT_EMBEDDER = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=torch.device(DEVICE))
TEXT_DECODER = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder", tokenizer="text_sonar_basic_decoder", device=torch.device(DEVICE))

def decode_embeddings(embeddings, lang):
    embeddings = embeddings.to(device=DEVICE, dtype=torch.float32)
    results = TEXT_DECODER.predict(
        embeddings, 
        target_lang=lang
    )
    return results

def get_eos_vector(lang):
    vec = TEXT_EMBEDDER.predict(["End of text."], source_lang=lang)
    if isinstance(vec, list):
        vec = vec[0]
    return torch.tensor(vec).squeeze().to(DEVICE).to(torch.float16)

def load_two_tower_model(checkpoint_path, device="cuda"):
    print(f"Loading model configuration...")
    config = two_tower_diffusion_lcm_1_6B()

    print("Building model structure...")
    model = create_two_tower_diffusion_lcm_model(
        config, 
        device=torch.device(device),
        dtype=torch.float16 
    )

    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    if "model" in state_dict:
        state_dict = state_dict["model"]
    
    model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    model.to(device).to(torch.float16)
    print("Model loaded successfully.")
    return model

def load_two_tower_model_7B(checkpoint_path, device="cuda"):
    print(f"Loading model configuration...")
    config = two_tower_diffusion_lcm_7B()

    print("Building model structure...")
    model = create_two_tower_diffusion_lcm_model(
        config, 
        device=torch.device(device),
        dtype=torch.float16 
    )

    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    if "model" in state_dict:
        state_dict = state_dict["model"]
    
    model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    model.to(device).to(torch.float16)
    print("Model loaded successfully.")
    return model

def run_inference_batched(generator, prompts_list):
    """
    Runs batched inference using the TwoTowerDiffusionLCMGenerator.
    """
    padded_batch = EmbeddingsBatch(*pad_seqs(prompts_list))
    output = generator(padded_batch, max_gen_len=1, min_gen_len=1, disable_cache=False) 
    return output

if __name__ == "__main__":
    
    if not Path(CHECKPOINT_PATH).exists():
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}. Script will crash at load step.")
    
    def limit_sentences(x):
        return {
            "prompt_sentences": x["text_sentences"][:1], 
            "target_sentences": x["text_sentences"][1:9]
        }

    if "7B" in CHECKPOINT_PATH:
        print("7B")
        model = load_two_tower_model_7B(CHECKPOINT_PATH, DEVICE)
    else:
        print("1_6B")
        model = load_two_tower_model(CHECKPOINT_PATH, DEVICE)
    
    options = DiffusionLCMGeneratorOptions(
        eos_threshold=0.9,
        inference_timesteps=40,
        initial_noise_scale=0.6,
        guidance_scale=3.0,
        guidance_rescale=0.7,
        epsilon_scaling=1.00045,
        stop_on_repetition_cosine_threshold=0.9,
        seed=42
    )

    for lang in LANGS:
        
        if os.path.isfile(f"results_{DS}_{lang}_{CHECKPOINT_PATH.replace('/', '_')}.jsonl"):
            print(f"SKIP results_{DS}_{lang}_{CHECKPOINT_PATH.replace('/', '_')}.jsonl")
            continue

        generator = TwoTowerDiffusionLCMGenerator(model, options, eos_vec=get_eos_vector(lang))

        if DS == "wiki":
            ds = load_dataset("parquet", data_files={"train": f"data/google_processed_wiki/{lang}.parquet"}, cache_dir="./cache")["train"]
        elif DS == "c4":
            ds = load_dataset("parquet", data_files={"train": f"data/c4_processed/{lang}.parquet"}, cache_dir="./cache")["train"]
        elif DS == "eurlex":

            try:
                ds = load_dataset("parquet", data_files={"train": f"data/multi_eurlex_processed/{lang}.parquet"}, cache_dir="./cache")["train"]
            except Exception as e:
                print(e)
                continue

        ds = ds.filter(lambda x: len(x["text_sentences"]) >= 9)
        ds = ds.map(limit_sentences, num_proc=4)
        ds = ds.select(range(min(1000, len(ds)))) 
        
        with torch.no_grad():
            with open(f"results_{DS}_{lang}_{CHECKPOINT_PATH.replace('/', '_')}.jsonl", "w", encoding="utf8") as f_out:

                batch_prompts_text = []
                batch_targets_text = []

                def process_accumulated_batch():
                    if not batch_prompts_text:
                        return
                    
                    batch_size = len(batch_prompts_text)
                    
                    # Flatten the lists of sentences for bulk embedding 
                    flat_prompts = [sent for doc in batch_prompts_text for sent in doc]
                    flat_targets = [sent for doc in batch_targets_text for sent in doc]
                    
                    # 1. Embed all prompt and target sentences at once
                    prompt_embs_list = TEXT_EMBEDDER.predict(flat_prompts, source_lang=lang, batch_size=512)
                    target_embs_list = TEXT_EMBEDDER.predict(flat_targets, source_lang=lang, batch_size=512)
                    
                    def to_tensor(embs_list):
                        if isinstance(embs_list, list):
                            return torch.vstack(embs_list).to(DEVICE)
                        elif not isinstance(embs_list, torch.Tensor):
                            return torch.tensor(embs_list).to(DEVICE)
                        return embs_list.to(DEVICE)

                    # Reshape them back to (Batch_Size, 8 sentences, Emb_Dim)
                    prompt_embs = to_tensor(prompt_embs_list).view(batch_size, 1, -1)
                    target_embs = to_tensor(target_embs_list).view(batch_size, 8, -1)
                    
                    # Initialize prefixes with the prompt sentences
                    current_prefixes = [prompt_embs[i] for i in range(batch_size)]

                    generated_texts_history = [[] for _ in range(batch_size)]
                    generated_embs_history = [[] for _ in range(batch_size)] 
                    metrics_history = [[] for _ in range(batch_size)]

                    # 2. Autoregressive loop for 8 completion steps
                    for step in range(8):
                        
                        # Predict only the next single sentence using the global generator
                        results = run_inference_batched(generator, current_prefixes)
                        
                        next_embs = []
                        for idx, hyp in enumerate(results.hypotheses):
                            prompt_len = current_prefixes[idx].size(0)
                            
                            # Get the newly generated embedding
                            next_emb = hyp[0].seq[prompt_len] 
                            next_embs.append(next_emb)
                            
                        # Stack to (Batch_Size, Emb_Dim) for decoding
                        next_embs_tensor = torch.stack(next_embs)
                        step_embs_list = next_embs_tensor.cpu().tolist()
                        
                        # Decode the newly generated embeddings to text
                        step_generated_texts = decode_embeddings(next_embs_tensor, lang)
                        
                        # Re-embed generated text to evaluate the similarity to GT
                        re_embedded_raw = TEXT_EMBEDDER.predict(step_generated_texts, source_lang=lang, batch_size=512)
                        re_embedded_tensor = to_tensor(re_embedded_raw)

                        # Calculate metrics against the GT target specifically for this step
                        for j in range(batch_size):
                            gt_emb = target_embs[j, step].unsqueeze(0)
                            curr_next_emb = next_embs_tensor[j].unsqueeze(0)
                            curr_re_emb = re_embedded_tensor[j].unsqueeze(0)

                            l2_pred_vs_gt = (curr_next_emb - gt_emb).pow(2).sum(dim=-1).sqrt().item()
                            l2_decoded_vs_gt = (curr_re_emb - gt_emb).pow(2).sum(dim=-1).sqrt().item()

                            # Ratio of embeddings in the batch further away from pred than the GT
                            # Excludes self (index j)
                            other_batch_items = torch.cat([target_embs[:j], target_embs[j+1:]]) 
                            
                            # Flatten to: ((Batch-1)*8, Dim)
                            distractors_flat = other_batch_items.view(-1, gt_emb.size(-1)) 
                            
                            # Calculate distance to all distractors. 
                            dist_to_distractors = (distractors_flat - curr_next_emb).pow(2).sum(dim=-1).sqrt()
                            
                            if len(dist_to_distractors) > 0:
                                ca_score = (dist_to_distractors > l2_pred_vs_gt).float().mean().item()
                            else:
                                ca_score = 1.0 # Fallback if batch_size is 1

                            # Max cosine sim to context / Max cosine sim of GT to context
                            context_embs = current_prefixes[j] # Shape: (Context_Len, Dim)
                            
                            cs_pred_context = F.cosine_similarity(curr_next_emb, context_embs, dim=-1)
                            max_cs_pred = cs_pred_context.max().item()
                            
                            cs_gt_context = F.cosine_similarity(gt_emb, context_embs, dim=-1)
                            max_cs_gt = cs_gt_context.max().item()
                            
                            # Guard against division by zero
                            par_score = (max_cs_pred / max_cs_gt) if abs(max_cs_gt) > 1e-6 else 0.0

                            generated_texts_history[j].append(step_generated_texts[j])
                            generated_embs_history[j].append(step_embs_list[j])
                            
                            metrics_history[j].append({
                                "step": step,
                                "l2": l2_pred_vs_gt, 
                                "l2_roundtrip": l2_decoded_vs_gt,
                                "ca": ca_score,
                                "par": par_score
                            })

                        # Update prefixes for the next step with the actual ground truth 
                        for idx in range(batch_size):
                            current_prefixes[idx] = torch.cat([current_prefixes[idx], target_embs[idx][step].unsqueeze(0)], dim=0)

                    for j in range(batch_size):
                        record = {
                            "prompt": batch_prompts_text[j],
                            "gt": batch_targets_text[j],
                            "gt_embs": target_embs[j].cpu().tolist(),
                            "generated": generated_texts_history[j],
                            "generated_embs": generated_embs_history[j],
                            "metrics": metrics_history[j]
                        }
                        json.dump(record, f_out)
                        f_out.write('\n')
                    
                    # Clear for the next batch
                    batch_prompts_text.clear()
                    batch_targets_text.clear()

                for item in tqdm(ds):
                    batch_prompts_text.append(item["prompt_sentences"])
                    batch_targets_text.append(item["target_sentences"])
                    
                    if len(batch_prompts_text) >= TARGET_BATCH_SIZE:
                        process_accumulated_batch()
                
                # Process any remaining instances
                process_accumulated_batch()