import os
import csv
import transformers
import torch
#from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

import pandas as pd
import random
import numpy as np
#from common import LMForwardAPI
import argparse
from common_gpt import constrainScoreByWholeExact, pmi, ApiCallLimitError, CompleteGPT, get_answer
from common_gpt import evaluateGPT3 as evaluate, testGPT3 as test
import re
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Pick Scores")
    parser.add_argument(
        "--task", 
        type=str, 
        required=True,
        default="cnn",
        help="cnn or gsm" 
    )
    parser.add_argument('--zero_shot', 
        action="store_true"
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
    )
    parser.add_argument(
        "--output_csv_file", 
        type=str, 
        required=True, 
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=14, 
    )
    parser.add_argument(
        "--mu", 
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--intrinsic_dim",
          type=int, default=10, 
          help="Intrinsic dimension for z_t.")
    
    parser.add_argument(
        "--n_prompt_tokens",
          type=int, default=5, 
          )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=0.2, 
        help="Learning rate for optimization.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on, e.g., 'cuda' or 'cpu'."
    )
    parser.add_argument(
        "--random_proj",
        type=str,
        default="uniform",
        choices=["normal", "uniform"],
        help="Random projection initialization: 'normal' or 'uniform'."
    )
    parser.add_argument(
        "--window_size",
        type=int, default=0, 
        help="Window size")
    
    parser.add_argument(
        "--alpha",
        type=float, default=0, 
        help="Weight for window_1.")
    parser.add_argument(
        "--beta",
        type=float, default=0, 
        help="Weight for window_2.")
    parser.add_argument(
        "--eli",
        type=float, default=1e-8, 
        help="Weight for window.")
    parser.add_argument(
        "--use_confident",
        type=int, default=0, 
        )
    
    parser.add_argument("--model_name_or_path", type=str, default="gpt-4", choices=["gpt-5", "gpt-4", "gpt-4o", "gpt-4o-mini"], help="Path to pretrained model or model identifier from huggingface.co/models.")

    return parser.parse_args()
    
args = parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    if args.task == "gsm":
        inputs = data['input'].tolist()
        labels = data['labels'].tolist()
    if args.task == "cnn":
        inputs = data['article'].tolist()
        labels = data['abstract'].tolist()
    if args.task == "wmt14":
        inputs = data['source_en'].tolist()
        labels = data['target_de'].tolist()
    return inputs, labels



def generate_answer_with_gpt(article):
    if args.task == "gsm":
        input_text = [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"Please provide a answer of the following math question. Place the finally result number after ' #### ' at the end of answer, for example:' #### 0' if the finally result number is '0'."
                                        ),
                                    }
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"\nQuestion:{article}\nAnswer:",
                                    }
                                ],
                            },
                    ]

    if args.task == "cnn":
        input_text = [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Please provide a summary of the following news."
                                    ),
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"\nNews:{article}\nSummary:",
                                }
                            ],
                        },
                    ]
    if args.task == "wmt14":
        input_text = [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Please translate the following English sentence into German."
                                    ),
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"\nEnglish:{article}\nGerman:",
                                }
                            ],
                        },
                    ]

    completeGPT = CompleteGPT()
    response = completeGPT.train_api_request(input_text, max_tokens=512, model_name=args.model_name_or_path)
    return response


def generate_summary(article):
    generated_text_final = generate_answer_with_gpt(article)
    return generated_text_final


def extract_equations(text):

    equations = re.findall(r"[0-9+\-*/=<>]+", text)
    return set(equations)

def logic_matching_loss(reference_text, generated_text):

    reference_equations = extract_equations(reference_text)
    generated_equations = extract_equations(generated_text)
    intersection = len(reference_equations & generated_equations)
    union = len(reference_equations | generated_equations)
    similarity = intersection / union if union > 0 else 0
    loss = 1 - similarity
    return loss


def get_answer(response):
    # answer_start_index = response.find("ANSWER:")
    # if answer_start_index == -1:
    #     response = response.strip()  
    # else:
    #     answer_start_index += len("ANSWER:")
    #     response = response[answer_start_index:].strip()
    match = re.search(r'####\s(\d+)', response)
    # match = re.search(r"####\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(\.\d+)?)$", response)

    #equations = re.findall(r"####\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(\.\d+)?)", text)
    
    if match:
        numeric_string = match.group(1)
        numeric_value = float(numeric_string.replace(',', ''))
        return numeric_value
    else:
        return None
    

def calculate_rouge1_loss(reference_text, generated_text):
    if not generated_text or not reference_text:
        return 1.0  
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    rouge1_f1 = scores['rouge1'].fmeasure  
    return 1 - rouge1_f1

def calculate_sentence_bleu_loss(reference_text, generated_text):
    if not generated_text or not reference_text:
        return 1.0
    ref_tokens = reference_text.split()
    gen_tokens = generated_text.split()
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu(
        [ref_tokens],
        gen_tokens,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothie
    )
    return 1 - bleu_score



def process_and_record_loss_to_csv(input_directory, output_csv_file):
    b=0
    Loss=0
    with open(output_csv_file, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["file", "loss"])
        article, reference_abstract = prepare_data_from_csv(input_directory)
        for data,answer in zip(article,reference_abstract):
            generated_abstract = generate_summary(data)
            b=b+1
            if args.task == "gsm":
                reference_answer_value = get_answer(answer)
                generated_answer_value = get_answer(generated_abstract)
                if reference_answer_value is None or generated_answer_value is None:
                    print(f"Skipping due to invalid answers. Question: {data}")
                    loss = 1
                else:
                    loss = 0 if reference_answer_value == generated_answer_value else 1
                print(f"File: {b}, Loss: {loss:.4f}")
                Loss = Loss + loss
                csv_writer.writerow([
                    b, 
                    # reference_abstract.replace("\n", " ").strip(), 
                    # generated_abstract.replace("\n", " ").strip(), 
                    loss
                ])
                if b >= 500:
                    csv_writer.writerow([
                    "Average Loss", 
                    Loss/b
                ])
                    break
            elif args.task == "cnn":
                loss = calculate_rouge1_loss(answer, generated_abstract)
                print(f"File: {b}, Loss: {loss:.4f}")
                Loss = Loss + loss
                csv_writer.writerow([
                    b,  
                    loss
                ])
                if b >= 500:
                    csv_writer.writerow([
                    "Average Loss", 
                    Loss/b
                ])
                    break
            elif args.task == "wmt14":
                loss = calculate_sentence_bleu_loss(answer, generated_abstract)
                print(f"File: {b}, Loss: {loss:.4f}")
                Loss = Loss + loss
                csv_writer.writerow([
                    b,  
                    loss
                ])
                if b >= 500:
                    csv_writer.writerow([
                    "Average Loss", 
                    Loss/b
                ])
                    break
            
    print(f"Loss records saved to {output_csv_file}")

def main():
    set_seed(args.seed)
    input_directory = args.input_file
    output_csv_file = args.output_csv_file
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    process_and_record_loss_to_csv(input_directory, output_csv_file)
    df = pd.read_csv(output_csv_file)
    average_loss = df['loss'].mean()
    print("Average loss:", average_loss)


if __name__ == "__main__":
    main()