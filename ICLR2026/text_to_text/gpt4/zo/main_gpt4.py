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

model_name = "lmsys/vicuna-7b-v1.5"
vicuna_model_path = "/hy-tmp/ICLR2026/text_to_text/models"
vicuna_tokenizer =transformers.AutoTokenizer.from_pretrained(model_name, cache_dir = vicuna_model_path)
vicuna_model = transformers.AutoModelForCausalLM.from_pretrained(model_name, cache_dir = vicuna_model_path)
vicuna_model = vicuna_model.to(args.device)


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

def text_to_embedding(text, tokenizer, model):
    input_token = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    emb_layer = model.get_input_embeddings()
    device = emb_layer.weight.device
    input_ids = input_token.input_ids.to(device)
    attention_mask = input_token.attention_mask.to(device)
    input_embed = emb_layer(input_ids)
    input_embed = input_embed.to(torch.float32)
    attention_mask = attention_mask.to(torch.float32)

    return input_embed, attention_mask


def generate_answer_with_gpt(instruction, article):
    instruction=instruction[0]
    if instruction.startswith("Prompt:"):
        instruction = instruction[len("Prompt:"):].strip()
    # print(instruction)
    if args.task == "gsm":
        input_text = [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"Instruction:{instruction} \n"
                                            "Please give an answer of the following math question according to the instruction. Place the finally result number after ' #### ' at the end of answer, for example:' #### 0' if the finally result number is '0'."
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
                                        f"Instruction:{instruction} \n"
                                        "Please give a summary of the following news according to the instruction."
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
                                        f"Instruction:{instruction} \n"
                                        f"Please translate the following English sentence into German according to the instruction."
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

def initialize_projection_matrix(intrinsic_dim, output_dim=None, random_proj="uniform", alpha=1.0, sigma=1.0, device="cpu", embedding_weights=None):
    embedding_dim = vicuna_model.get_input_embeddings().weight.shape[-1]
    A = torch.nn.Linear(args.intrinsic_dim, args.n_prompt_tokens * embedding_dim, bias=False).to(args.device)

    if random_proj == "normal":
        if embedding_weights is not None:
            mu_hat = embedding_weights.mean().item()
            std_hat = embedding_weights.std().item()
        else:
            mu_hat = 0.0
            std_hat = 1.0 
        mu = 0.0
        std = alpha * std_hat / (np.sqrt(intrinsic_dim) * sigma)

        print(f"[Embedding] mu: {mu_hat}, std: {std_hat} [RandProj] mu: {mu}, std: {std}")

        torch.nn.init.normal_(A.weight, mean=mu, std=std)

    elif random_proj == "uniform":
        torch.nn.init.uniform_(A.weight, -1, 1)

    else:
        raise ValueError(f"Unknown random_proj type: {random_proj}")

    print(f"A weight mean: {A.weight.mean().item()}, std: {A.weight.std().item()}")
    
    return A


def generate_summary(A,window_alpha,window_beta, window_dev,reference_abstract,article, z_t, mu):
    llama_system_prompt = (
        "Rewrite the provided example prompt to get better output.")

    if args.task == "gsm":
        template = (
            "{llama_system_prompt}\n"
            "Example Prompt: Solve the problem logically and step by step, making sure every step is correct.\n"
            "User: Directly output prompt content. \n"
            "Assistant:"
        )

    elif args.task == "cnn":
        template = (
            "{llama_system_prompt}\n"
            "Example Prompt: Summarize the following article as a concise, factual news brief. Preserve all key entities, events, and outcomes. Avoid unnecessary details.\n"
            "User: Directly output prompt without adding any additional content. \n"
            "Assistant:"
        )

    elif args.task == "wmt14":
        template = (
            "{llama_system_prompt}\n"
            "Example Prompts: You are a professional translator specializing in English to German translation. Translate the following English sentence to German accurately and naturally, maintaining the original meaning and tone.\n"
            "User: Directly output prompt words. \n"
            "Assistant:"
        )


    # template = template.replace("[question]", article)
    # template = template.replace("[output]", reference_abstract)
    template = template.replace("{llama_system_prompt}", llama_system_prompt)
    input_embed, attention_mask = text_to_embedding(template, vicuna_tokenizer, vicuna_model)
    input_embed = input_embed.to(z_t.device)
    attention_mask = attention_mask.to(z_t.device)

    #A = torch.nn.Parameter(torch.randn(input_embed.shape[-1], args.intrinsic_dim).to(z_t.device) * 0.01)
    #A = torch.nn.Linear(args.intrinsic_dim, args.n_prompt_tokens * input_embed.shape[-1], bias=False).to(args.device)


    '''
    if random_proj == 'normal':
            # calculate std for normal distribution
            if model_name in ['wizardlm', 'vicuna', 'openchat']:
                print('Get the embedding firstly to avoid issues')
            else:
                raise NotImplementedError
            mu_hat = self.embedding.reshape(-1).mean().item()
            std_hat = self.embedding.reshape(-1).std().item()
            mu = 0.0
            std = args.alpha * std_hat / (np.sqrt(intrinsic_dim) * args.sigma)

            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            torch.nn.init.normal_(self.linear.weight, mu, std)
        elif random_proj == 'uniform':  
            torch.nn.init.uniform_(self.linear.weight, -1, 1)

    '''
    attention_mask_extended = torch.cat(
            (torch.ones(attention_mask.shape[0], args.n_prompt_tokens).to(attention_mask.device), attention_mask), dim=1
            )

    if not args.zero_shot:
        z_tem = z_t.unsqueeze(0)
        noise = torch.normal(mean=0.0, std=1.0, size=z_tem.shape).to(z_t.device)
        noise = noise/torch.norm(noise, dim=1, keepdim=False)
        z_t_pos = z_tem + mu * noise
        z_t_neg = z_tem - mu * noise
        Az_pos = A(z_t_pos)   
        Az_neg = A(z_t_neg)  

        #input_embed_pos = torch.cat((input_embed[:, :1, :], Az_pos.unsqueeze(1), input_embed[:, 1:, :]), dim=1)
        #input_embed_neg = torch.cat((input_embed[:, :1, :], Az_neg.unsqueeze(1), input_embed[:, 1:, :]), dim=1)

        Az_pos_reshaped = Az_pos.view(input_embed.shape[0], args.n_prompt_tokens, input_embed.shape[-1])
        Az_neg_reshaped = Az_neg.view(input_embed.shape[0], args.n_prompt_tokens, input_embed.shape[-1])

        #input_embed_pos = torch.cat((Az_pos_reshaped, input_embed), dim=1)
        #input_embed_neg = torch.cat((Az_neg_reshaped, input_embed), dim=1)###
        input_embed_pos = torch.cat((Az_pos_reshaped, input_embed), dim=1)
        input_embed_neg = torch.cat((Az_neg_reshaped, input_embed), dim=1)###

        if attention_mask_extended.shape[1] != input_embed_pos.shape[1]:
            attention_mask_extended = attention_mask_extended[:, :input_embed_pos.shape[1]]

        #attention_mask_extended = torch.cat(
        #    (attention_mask, torch.ones(attention_mask.shape[0], z_t_pos.shape[1]).to(attention_mask.device)), dim=1
        #)

        
        instruction_pos = vicuna_model.generate(
        inputs_embeds = input_embed_pos,
        attention_mask = attention_mask_extended,
        max_new_tokens = 256)
        instruction_neg = vicuna_model.generate(
        inputs_embeds = input_embed_neg,
        attention_mask = attention_mask_extended,
        max_new_tokens = 256)
        instruction_pos = vicuna_tokenizer.batch_decode(instruction_pos, skip_special_tokens=True)
        instruction_neg = vicuna_tokenizer.batch_decode(instruction_neg, skip_special_tokens=True)

        generated_text_pos = generate_answer_with_gpt(instruction_pos,article)
        generated_text_neg = generate_answer_with_gpt(instruction_neg,article)

        if args.task == "gsm":
            loss_pos = logic_matching_loss(reference_abstract, generated_text_pos)
            loss_neg = logic_matching_loss(reference_abstract, generated_text_neg)
        if args.task == "cnn":
            loss_pos = calculate_rouge1_loss(reference_abstract, generated_text_pos)
            loss_neg = calculate_rouge1_loss(reference_abstract, generated_text_neg)
        if args.task == "wmt14":
            loss_pos = calculate_sentence_bleu_loss(reference_abstract, generated_text_pos)
            loss_neg = calculate_sentence_bleu_loss(reference_abstract, generated_text_neg)

        loss_pos = torch.tensor(loss_pos, dtype=torch.float32, device=z_t.device)
        loss_neg = torch.tensor(loss_neg, dtype=torch.float32, device=z_t.device)
        
        g_t_hat = torch.mean(((loss_pos - loss_neg).reshape(-1, 1) * noise) / (2 * mu), dim=0)
        if args.alpha !=0:
            total_g_t = torch.cat([window_dev, g_t_hat.unsqueeze(0)], dim=0)
            total_g_t = total_g_t[1:, :]
            g_t_finally = torch.matmul(total_g_t.T, window_alpha).squeeze(1)
            g_t_finally = g_t_finally/torch.sum(window_alpha, dim=0, keepdim=False)
            if args.use_confident==1:   
                sign_match_mask = (g_t_finally * g_t_hat) > 0
                g_t_finally = torch.where(sign_match_mask, g_t_finally, torch.zeros_like(g_t_finally))
            if args.beta !=0:
                g_squared = total_g_t ** 2
                g_t_finally_B = torch.matmul(g_squared.T, window_beta).squeeze(1)
                g_t_finally_B = g_t_finally_B/torch.sum(window_beta, dim=0, keepdim=False)
                g_t_finally_B = torch.sqrt(g_t_finally_B)+torch.full_like(g_t_finally_B, args.eli)
            if args.beta==0:
                g_t_finally_B=1
        if args.alpha==0:
            g_t_finally=g_t_hat
            g_t_finally_B=1
            total_g_t=0
        z_t = z_t - args.learning_rate * g_t_finally/g_t_finally_B 
 
    z_tem = z_t.unsqueeze(0)
    z_t_projected = A(z_tem)  
    z_t_projected = z_t_projected.view(1, args.n_prompt_tokens, input_embed.shape[-1])  
    input_embed_final = torch.cat((z_t_projected, input_embed), dim=1)
    input_embed_final = vicuna_model.generate(
    inputs_embeds = input_embed_final,
    attention_mask = attention_mask_extended,
    max_new_tokens = 256)
    input_embed_final = vicuna_tokenizer.batch_decode(input_embed_final, skip_special_tokens=True)
    if args.zero_shot:
        total_g_t = window_dev
    generated_text_final = generate_answer_with_gpt(input_embed_final, article)
    return generated_text_final, z_t, total_g_t


def extract_equations(text):

    equations = re.findall(r"[0-9+\-*/=<>]+", text)
    return set(equations)

def logic_matching_loss(reference_text, generated_text):
    if not isinstance(generated_text, str):
        generated_text = str(generated_text)
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
    z_t = torch.zeros(args.intrinsic_dim).type(torch.float32).to(args.device)
    A = initialize_projection_matrix(args.intrinsic_dim, output_dim=None, random_proj="uniform", alpha=1.0, sigma=1.0, device="cpu", embedding_weights=None)
    b=0
    Loss=0
    window_dev =  torch.zeros((args.window_size, args.intrinsic_dim)).to(args.device)
    window_alpha = torch.tensor([args.alpha**(args.window_size-1-i) for i in range(args.window_size)]).unsqueeze(1).to(args.device)
    window_beta = torch.tensor([args.beta**(args.window_size-1-i) for i in range(args.window_size)]).unsqueeze(1).to(args.device)
    with open(output_csv_file, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["file", "loss"])
        article, reference_abstract = prepare_data_from_csv(input_directory)
        for data,answer in zip(article,reference_abstract):
            generated_abstract, z_, window_dev = generate_summary(A,window_alpha,window_beta, window_dev, answer,data, z_t, mu=args.mu)
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
            elif args.task == "wmt14":
                loss = calculate_sentence_bleu_loss(answer, generated_abstract)
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