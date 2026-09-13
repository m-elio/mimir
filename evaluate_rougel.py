import os
import json
import argparse
import evaluate

parser = argparse.ArgumentParser(description="Evaluate translation JSON files using chrF++ and COMET.")
parser.add_argument("--results_path", type=str, required=True, 
                    help="Path to the JSON file containing your inference results.")
args = parser.parse_args()

_MBERT = None
 
def multilingual_tokenize(text: str):
    global _MBERT
    if _MBERT is None:
        from transformers import AutoTokenizer
        _MBERT = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
    return _MBERT.tokenize(text.lower())


def evaluate_json_rouge_hf(file_path):

    rouge_metric = evaluate.load('rouge')

    with open("results_it.jsonl", "w", encoding="utf8") as f_out:

        for x in sorted(os.listdir(file_path)):

            print(x)
        
            predictions = []
            references = []
            
            try:

                data = []
                with open(os.path.join(file_path, x), 'r', encoding='utf-8') as file:

                    for l in file:
                        line_data = json.loads(l)
                        if "generated" not in line_data or "gt" not in line_data:
                            print("SKIP")
                            continue
                        data.append(line_data)
                    
                    if isinstance(data, dict):
                        data = [data]
                        
                    for item in data:
                        predictions.append(item["generated"])
                        references.append(item["gt"])
                
                results = rouge_metric.compute(
                    predictions=predictions, 
                    references=references, 
                    use_stemmer=False,
                    tokenizer=multilingual_tokenize,
                )
                
                print("-" * 25)
                print(f"ROUGE-1: {results.get('rouge1', 0):.4f}")
                print(f"ROUGE-2: {results.get('rouge2', 0):.4f}")
                print(f"ROUGE-L: {results.get('rougeL', 0):.4f}")
                print("-" * 25)

                dataset = x.split("_")[1]
                lang = x.split("_")[2] + "_" + x.split("_")[3]

                if "__" in x:
                    model = x.split("__")[-1].replace('.jsonl', '')
                else:
                    model = x.split("_")[-1].replace('.jsonl', '')

                new_dict = {}
                new_dict["ds"] = dataset
                new_dict["model"] = model
                new_dict["lang"] = lang
                new_dict["rougeL"] = results['rougeL']

                json.dump(new_dict, f_out)
                f_out.write('\n')

            except Exception as e:
                print(e)


if __name__ == "__main__":
    evaluate_json_rouge_hf(args.results_path)