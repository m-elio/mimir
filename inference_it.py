import os
import json
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
parser.add_argument('-g', '--guidance_scale', type=float, required=True)
args = parser.parse_args()

ASSISTANT_TRANSLATION = {
    "arb_Arab": "دور المساعد.", 
    "bel_Cyrl": "Памочнік, свой ход.", 
    "ben_Beng": "সহকারী পালা।",
    "bos_Latn": "Pomoćni potez.", 
    "bul_Cyrl": "Ред е на помощника.", 
    "cat_Latn": "Torn de l'ajudant.",
    "ces_Latn": "Na řadě je asistent.", 
    "cym_Latn": "Tro'r cynorthwyydd.", 
    "dan_Latn": "Assistenten er på tur.",
    "deu_Latn": "Der Assistent ist an der Reihe.", 
    "eng_Latn": "Assistant turn.", 
    "fra_Latn": "C'est au tour de l'assistant.",
    "heb_Hebr": "תורו של העוזר.", 
    "hin_Deva": "सहायक की बारी।", 
    "hrv_Latn": "Pomoćni potez.",
    "ind_Latn": "Giliran asisten.", 
    "jpn_Jpan": "アシスタントの番。", 
    "ita_Latn": "È il turno dell'assistente.",
    "kan_Knda": "ಸಹಾಯಕ ತಿರುವು.", 
    "kor_Hang": "조수 차례.", 
    "lvs_Latn": "Palīga kārta.",
    "mal_Mlym": "സഹായിയുടെ ഊഴം",
    "mar_Deva": "सहाय्यक पालवी.",
    "mkd_Cyrl": "Помошник-ред.",
    "npi_Deva": "सहायक पालो।",
    "nld_Latn": "De assistent is aan de beurt.",
    "ory_Orya": "ଆସିଷ୍ଟାଣ୍ଟ ଟର୍ନ୍ |",
    "pol_Latn": "Rzuty pomocników.",
    "por_Latn": "É a vez do assistente.",
    "ron_Latn": "E rândul asistentului.",
    "rus_Cyrl": "Очередь помощника.",
    "slk_Latn": "Na rade je asistent.",
    "slv_Latn": "Na vrsti je pomočnik.",
    "srp_Cyrl": "Помоћни круг.",
    "spa_Latn": "Turno del asistente.",
    "swe_Latn": "Assistenten är på tur.",
    "swh_Latn": "Mzunguko wa msaidizi.",
    "tam_Taml": "துணைவரின் முறை.",
    "tel_Telu": "సహాయకుడి వంతు",
    "tha_Thai": "ผู้ช่วยเลี้ยว",
    "tur_Latn": "Sıra asistanında.",
    "ukr_Cyrl": "Черга помічника.",
    "urd_Arab": "معاون کی باری۔",
    "vie_Latn": "Đến lượt trợ lý.",
    "zho_Hans": "助手的回合。"
}

USER_TRANSLATION = {
    "arb_Arab": "دور المستخدم.", 
    "bel_Cyrl": "Ход карыстальніка.", 
    "ben_Beng": "ব্যবহারকারীর পালা।",
    "bos_Latn": "Red korisnika.", 
    "bul_Cyrl": "Ред е на потребителя.", 
    "cat_Latn": "Torn de l'usuari.",
    "ces_Latn": "Je řada na uživateli.", 
    "cym_Latn": "Tro'r defnyddiwr.", 
    "dan_Latn": "Brugerens tur.",
    "deu_Latn": "Der Benutzer ist am Zug.", 
    "eng_Latn": "User turn.", 
    "fra_Latn": "C'est au tour de l'utilisateur.",
    "heb_Hebr": "תור המשתמש.", 
    "hin_Deva": "उपयोगकर्ता की बारी।", 
    "hrv_Latn": "Poteg korisnika.",
    "ind_Latn": "Giliran pengguna.", 
    "jpn_Jpan": "ユーザーの番です。", 
    "ita_Latn": "È il turno dell'utente.",
    "kan_Knda": "ಬಳಕೆದಾರರ ತಿರುವು.", 
    "kor_Hang": "사용자 차례입니다.", 
    "lvs_Latn": "Lietotāja kārta.",
    "mal_Mlym": "ഉപയോക്താവിന്റെ ഊഴം",
    "mar_Deva": "वापरकर्त्याची पाळी.",
    "mkd_Cyrl": "Потребен е корисник.",
    "npi_Deva": "प्रयोगकर्ताको पालो।",
    "nld_Latn": "De gebruiker is aan de beurt.",
    "ory_Orya": "ୟୁଜର୍ ଟର୍ନ୍ |",
    "pol_Latn": "Ruch użytkownika.",
    "por_Latn": "É a vez do utilizador.",
    "ron_Latn": "E rândul utilizatorului.",
    "rus_Cyrl": "Ход пользователя.",
    "slk_Latn": "Je na ťa.",
    "slv_Latn": "Na vrsti je uporabnik.",
    "srp_Cyrl": "Кориснички потез.",
    "spa_Latn": "Turno del usuario.",
    "swe_Latn": "Användarens tur.",
    "swh_Latn": "Mzunguko wa mtumiaji.",
    "tam_Taml": "பயனர் முறை.",
    "tel_Telu": "వినియోగదారుని వంతు",
    "tha_Thai": "ผู้ใช้เทิร์น",
    "tur_Latn": "Kullanıcı sırası.",
    "ukr_Cyrl": "Хід користувача.",
    "urd_Arab": "صارف کی باری۔",
    "vie_Latn": "Đến lượt người dùng.",
    "zho_Hans": "轮到用户了。"
}

LANGS = list(ASSISTANT_TRANSLATION.keys())
    
DS = args.ds
CHECKPOINT_PATH = args.checkpoint_path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_BATCH_SIZE = 512 

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

def run_inference_batched(model, lang, prompts_list, device="cuda"):
    """
    Runs batched inference using the TwoTowerDiffusionLCMGenerator.
    """
    options = DiffusionLCMGeneratorOptions(
        eos_threshold=0.9,
        inference_timesteps=40,
        initial_noise_scale=0.6,
        guidance_scale=args.guidance_scale,
        guidance_rescale=0.7,
        epsilon_scaling=1.00045,
        stop_on_repetition_cosine_threshold=0.9,
        seed=42
    )
    generator = TwoTowerDiffusionLCMGenerator(model, options, eos_vec=get_eos_vector(lang))
    padded_batch = EmbeddingsBatch(*pad_seqs(prompts_list))
    
    output = generator(padded_batch, max_gen_len=16, min_gen_len=1, disable_cache=False) 
    return output

if __name__ == "__main__":
    
    if not Path(CHECKPOINT_PATH).exists():
        print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}. Script will crash at load step.")

    if "7B" in CHECKPOINT_PATH:
        print("7B")
        model = load_two_tower_model_7B(CHECKPOINT_PATH, DEVICE)
    else:
        print("1_6B")
        model = load_two_tower_model(CHECKPOINT_PATH, DEVICE)
    
    os.makedirs(f"it_eval_{str(args.guidance_scale)}", exist_ok=True)

    for lang in LANGS:
        
        if os.path.isfile(f"it_eval_{str(args.guidance_scale)}/results_{DS}_{lang}_{CHECKPOINT_PATH.replace('/', '_')}.jsonl"):
            print(f"SKIP it_eval_{str(args.guidance_scale)}/results_{DS}_{lang}_{CHECKPOINT_PATH.replace('/', '_')}")
            continue
        
        if DS == "mlqa":

            try:
                ds = load_dataset("parquet", data_files={"train": f"data/mlqa_processed/{lang}.parquet"}, cache_dir="./cache")["train"]
            except Exception as e:
                print(e)
                continue
        elif DS == "xlsum":

            try:
                ds = load_dataset("parquet", data_files={"train": f"data/xl_sum_test_processed/{lang}.parquet"}, cache_dir="./cache")["train"]
            except Exception as e:
                print(e)
                continue
        elif DS == "xl_wsd":

            try:
                ds = load_dataset("parquet", data_files={"train": f"data/xl_wsd_processed/{lang}.parquet"}, cache_dir="./cache")["train"]
            except Exception as e:
                print(e)
                continue
            
        with torch.no_grad():
            with open(f"it_eval_{str(args.guidance_scale)}/results_{DS}_{lang}_{CHECKPOINT_PATH.replace('/', '_')}.jsonl", "w", encoding="utf8") as f_out:

                batch_prompts_text = []
                batch_targets_text = []

                def process_accumulated_batch(lang):
                    if not batch_prompts_text:
                        return

                    batch_size = len(batch_prompts_text)
                    
                    with torch.inference_mode():
                        # 1. Embed Prompts 
                        # TEXT_EMBEDDER handles list of lists (documents of sentences) automatically
                        doc_lengths = [len(doc) for doc in batch_prompts_text]
            
                        # 2. Flatten the nested list into a single list of strings
                        flat_prompts = [sentence for doc in batch_prompts_text for sentence in doc]
                        
                        # 3. Predict all embeddings in one highly efficient batched forward pass
                        # Returns a single tensor of shape: (total_sentences_in_batch, 1024)
                        flat_embs = TEXT_EMBEDDER.predict(flat_prompts, source_lang=lang).to(torch.float16)
                        
                        # 4. Split the flat tensor back into a list of individual document tensors
                        prompt_embs_list = list(torch.split(flat_embs, doc_lengths))

                        output = run_inference_batched(model, lang, prompt_embs_list)
                        
                        generated_texts_history = []
                        
                        # 4. Decode & Evaluate
                        for i, hyp_list in enumerate(output.hypotheses):

                            print(i)

                            # Take the top hypothesis
                            hyp = hyp_list[0]
                            
                            # Decode the continuous tensor back into text
                            # We wrap hyp.seq in a list so the pipeline processes it as 1 document
                            decoded_sentences = TEXT_DECODER.predict(
                                hyp.seq[prompt_embs_list[i].shape[0]:].to(torch.float32), 
                                target_lang=lang,
                            )

                            decoded_sentences = decoded_sentences[:-1]
                            
                            # Join generated sentences into a single string for ROUGE evaluation
                            if isinstance(decoded_sentences, list):
                                generated_text = " ".join(decoded_sentences)
                            else:
                                generated_text = decoded_sentences
                                
                            generated_texts_history.append(generated_text)

                        for j in range(batch_size):
                            record = {
                                "prompt": batch_prompts_text[j],
                                "gt": batch_targets_text[j],
                                "generated": generated_texts_history[j]
                            }
                            json.dump(record, f_out)
                            f_out.write('\n')
                            
                        # Clear for the next batch
                        batch_prompts_text.clear()
                        batch_targets_text.clear()

                for item in tqdm(ds):

                    if item["text_sentences"][-1] == "Answer:":
                        item["text_sentences"] = item["text_sentences"][:-1]

                    batch_prompts_text.append([USER_TRANSLATION[lang]] + item["text_sentences"] + [ASSISTANT_TRANSLATION[lang]])
                    batch_targets_text.append(item["label"])
                    
                    if len(batch_prompts_text) >= TARGET_BATCH_SIZE:
                        process_accumulated_batch(lang)
                
                # Process any remaining instances
                process_accumulated_batch(lang)