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
from rouge_score import rouge_scorer
import warnings
from transformers.utils import logging

warnings.filterwarnings("ignore", category=UserWarning)
logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Pick Scores")
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


vicuna_model_path = "/hy-tmp/ICLR2026/text_to_text/models/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"
vicuna_tokenizer = transformers.AutoTokenizer.from_pretrained(vicuna_model_path, local_files_only=True)
vicuna_model = transformers.AutoModelForCausalLM.from_pretrained(vicuna_model_path, local_files_only=True,device_map="auto").eval()#,torch_dtype=torch.float16
model_id = "/hy-tmp/ICLR2026/text_to_text/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    local_files_only=True 
)


pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def read_story_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    article, abstract = [], []
    is_highlight = False 

    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("@highlight"):
            is_highlight = True  
            continue  
        if is_highlight:
            abstract.append(line)
        else:
            article.append(line)
    return " ".join(article).strip(), " ".join(abstract).strip()


def text_to_embedding(text, tokenizer, model): 
    model.to(args.device)
    input_token = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(args.device)  
    input_ids = input_token.input_ids          
    attention_mask = input_token.attention_mask 

    input_embed = model.get_input_embeddings()(input_ids)
    input_embed = input_embed.to(torch.float32)
    attention_mask = attention_mask.to(torch.float32)

    return input_embed, attention_mask



def generate_answer_with_Qwen(instruction, model, tokenizer, article):
    instruction=instruction[0]
    instruction_index = instruction.find("Instructions:") + len("Instructions:")
    instruction = instruction[instruction_index:].strip()
    # print(instruction)
    # print(article)
    prompt = "\n\nInstructions:[instruction]\nPlease give a summary of the following news according to the instructions.\nNews:[article]\nSummary:"
    prompt = prompt.replace("[instruction]", instruction)
    prompt = prompt.replace("[article]", article)
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}]
    # print(messages)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs,max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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



def generate_summary(A,reference_abstract,article, z_t, tokenizer, model, mu):

    llama_system_prompt = (
        "You are a prompt designer. Your task is to create a new prompt based on the example provided. "
        "You are not allowed to perform the task itself. Instead, you must generate a prompt in the same style as the example. "
    )

    template = (
            "{llama_system_prompt}\n"
            "Example Prompt: \"Summarize the news article, preserving key facts and removing unnecessary details. Output only the summary.\"\n"
            "USER: Generate a new prompt for a news summarization task.\n"
            "ASSISTANT:"
        )
    # template = template.replace("[article]", article)
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

    z_t = z_t.unsqueeze(0) 
    z_t_projected = A(z_t)  

    z_t_projected = z_t_projected.view(1, args.n_prompt_tokens, input_embed.shape[-1]) 

    input_embed_final = torch.cat((z_t_projected, input_embed), dim=1)
 
    attention_mask_extended = torch.cat(
        (torch.ones(attention_mask.shape[0], args.n_prompt_tokens).to(attention_mask.device), attention_mask), dim=1
        )

    if attention_mask_extended.shape[1] != input_embed_final.shape[1]:
        attention_mask_extended = attention_mask_extended[:, :input_embed_final.shape[1]]
    input_embed_final = vicuna_model.generate(
    inputs_embeds = input_embed_final,
    attention_mask = attention_mask_extended,
    max_new_tokens = 256,
    do_sample=False
    )
    input_embed_final = vicuna_tokenizer.batch_decode(input_embed_final, skip_special_tokens=True)
    
    generated_text_final = generate_answer_with_Qwen(input_embed_final, model, tokenizer,article)

    #loss_final = calculate_rouge1_loss(reference_abstract, generated_text_final)
    
    return generated_text_final.strip(), z_t





def calculate_rouge1_loss(reference_text, generated_text):

    if not generated_text or not reference_text:
        return 1.0  

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    rouge1_f1 = scores['rouge1'].fmeasure  
    return 1 - rouge1_f1  

def process_and_record_loss_to_csv(input_directory, output_csv_file):


    z_t = torch.zeros(args.intrinsic_dim).type(torch.float32).to(args.device)
    A=initialize_projection_matrix(args.intrinsic_dim, output_dim=None, random_proj="uniform", alpha=1.0, sigma=1.0, device=args.device, embedding_weights=None)
    b=0
    Loss=0
    total_time=0
    with open(output_csv_file, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)

        csv_writer.writerow(["file", "loss"])
        df = pd.read_csv(input_directory)
        for idx, row in df.iterrows():
            article = row["article"]
            reference_abstract = row["abstract"]
            import time
            start_time = time.perf_counter() 
            b=b+1
            generated_abstract, z_ = generate_summary(A, reference_abstract,article, z_t, tokenizer, model, mu=args.mu)
            z_t = z_.squeeze(0)
            loss = calculate_rouge1_loss(reference_abstract, generated_abstract)
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
            end_time = time.perf_counter()
            iteration_time = end_time - start_time
            total_time += iteration_time
            average_time = total_time / b
            print(f"Iteration {b} took {average_time:.4f} seconds.")

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