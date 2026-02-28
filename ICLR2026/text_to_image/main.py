import argparse
from mydataset import testDataset, subsampled_data
from torch.utils.data import DataLoader
from common import LMForwardAPI, set_all_seed
import torch
from automatic_prompt_engineer import template
import os
from tqdm import tqdm
import copy
from PromptScorer import PromptScorer
import time
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="InstructZero pipeline")
    parser.add_argument(
        "--random_proj",
        type=str,
        default="uniform",
        help="The initialization of the projection matrix A."
    )
    parser.add_argument(
        "--intrinsic_dim",
        type=int,
        default=10,
        help="The instrinsic dimension of the projection matrix"
    )
    parser.add_argument(
        "--n_prompt_tokens",
        type=int,
        default=5,
        help="The number of prompt tokens."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set the seed."    
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Set the alpha if the initialization of the projection matrix A is std."    
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Set the sigma if the initialization of the projection matrix A is std."    
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='vicuna',
        help="The model name of the open-source LLM."    
    )
    parser.add_argument(
        "--HF_cache_dir",
        type=str,
        default="lmsys/vicuna-13b-v1.3",
        help="Your vicuna directory"
    )
    parser.add_argument(
        "--sdmodel_name", 
        type=str, 
        default="/root/autodl-tmp/model/diffusion_model/stable-diffusion-v1.5/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="/hy-tmp/ICLR2026/text_to_image/models/"
    )
    parser.add_argument(
        "--epoch", 
        type=int,
        default=1
    )
    parser.add_argument(
        "--mu", 
        type=float,
        default=0.01
    )
    parser.add_argument(
        "--q", 
        type=float,
        default=1,
        help="Number of samples of noise"
    )
    parser.add_argument(
        "--cuda", 
        type=int, 
        default=0
    )
    parser.add_argument(
        "--learning_rate", 
        type=float,
        default=0.1
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="./datasets/benchmark_anime_256.csv"
    )
    parser.add_argument(
        "--sample_file", 
        type=str, 
        default="./train_data/train_for_ft.txt"
    )
    parser.add_argument('--weights', 
        nargs='+',
        type=float,
        default=[1, 0, 0]
    )
    parser.add_argument('--ptype', 
        type=int,
        default=5
    )
    parser.add_argument('--example', 
        type=str,
        default='promptist'
    )
    parser.add_argument('--zero_shot', 
        action="store_true"
    )
    parser.add_argument('--start', 
        type=int,
        default=0
    )
    parser.add_argument('--number', 
        type=int,
        default=100000
    )
    
    parser.add_argument(
        "--window_size",
        type=int, default=10, 
        help="Window size")
    
    parser.add_argument(
        "--alpha_w",
        type=float, default=0.95, 
        help="Weight for window_1.")
    parser.add_argument(
        "--beta_w",
        type=float, default=0.99, 
        help="Weight for window_2.")
    parser.add_argument(
        "--eli",
        type=float, default=1e-8, 
        )
    parser.add_argument(
        "--use_confident",
        type=int, default=0, 
        )

    args = parser.parse_args()
    args.device = torch.device("cuda", args.cuda)

    return args


def run(args):
    test_dataset = testDataset(args.input_file)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    scorer = PromptScorer(args.sdmodel_name, args.cache_dir, args.device)

    demos_template = "Original: [INPUT]\nRephrase: [OUTPUT]"
    init_prompt = ['\n']
    d_template = template.DemosTemplate(demos_template)
    demos = d_template.fill(subsampled_data[args.example])
    # TODO: modify template
    # prompt_gen_template = "[full_DEMO]\n\nRephrase:"
    if args.ptype == 0:
        prompt_gen_template = "[full_DEMO]\n\nBased on the rephrasing way in the examples above, rephrase this sentence with consistency guaranteed:\n[original_prompt]\n"
    elif args.ptype == 1:
        prompt_gen_template = "[full_DEMO]\n\nBased on the rephrasing way in the examples above, rephrase this sentence:\n[original_prompt]\n"
    elif args.ptype == 2:
        prompt_gen_template = "[full_DEMO]\n\nBased on the rephrasing way in the examples above, using your creativity to rephrase this sentence:\n[original_prompt]\n"
    elif args.ptype == 3:
        prompt_gen_template = "[full_DEMO]\n\nIn order to make the diffusion model generate better pictures, based on the rephrasing way in the examples above, rephrase this sentence:\n[original_prompt]\n"
    elif args.ptype == 5:
        prompt_gen_template = "[full_DEMO]\n\nIn order to make the diffusion model generate better pictures, based on the rephrasing way in the examples above, using your creativity rather than just applying the example content to rephrase this sentence:\n[original_prompt]\n"
    prompt_gen_template = template.InitQATemplate(prompt_gen_template)
    
    model_forward_api = LMForwardAPI(model_name=args.model_name, init_prompt=init_prompt, prompt_gen_template=prompt_gen_template, random_proj=args.random_proj, 
                                    intrinsic_dim=args.intrinsic_dim, n_prompt_tokens=args.n_prompt_tokens, HF_cache_dir=args.HF_cache_dir, demos=demos, args=args)

    dirStr, ext = os.path.splitext(args.input_file)
    file = dirStr.split("/")[-1]
    ww = '_'.join(str(i) for i in args.weights)
    if "dreamlike" in args.sdmodel_name:
        model_identifier = "dreamlike"
    elif "v1-5" in args.sdmodel_name or "stable-diffusion-v1-5" in args.sdmodel_name:
        model_identifier = "v1.5"
    else:
        model_identifier = "unknown_model"
    if args.zero_shot:
        save_dir = f'./output_zs/{file}/{args.model_name}_{args.learning_rate}_{args.mu}_{model_identifier}/{args.seed}'
    else:
        save_dir = f'./output/{file}/{args.model_name}_{args.learning_rate}_{args.mu}_{args.window_size}_{args.alpha_w}_{args.beta_w}_{model_identifier}/{args.seed}'

    weights = args.weights
    window_dev =  torch.zeros((args.window_size, args.intrinsic_dim)).to(args.device)
    window_alpha = torch.tensor([args.alpha_w**(args.window_size-1-i) for i in range(args.window_size)]).unsqueeze(1).to(args.device)
    window_beta = torch.tensor([args.beta_w**(args.window_size-1-i) for i in range(args.window_size)]).unsqueeze(1).to(args.device)
    z_t = torch.zeros(args.intrinsic_dim).type(torch.float32).to(args.device)
    Sum_score = 0
    for batch_idx, batch_data in enumerate(test_dataloader):
        if not (batch_idx>=args.start and batch_idx<(args.start+args.number)):
            continue 
        if not args.zero_shot:
            # baseline_score = model_forward_api.cal_baseline(batch_data['text'], scorer)
            epoch_progress = tqdm(range(args.epoch), desc=f"Examples {batch_idx+1}/{len(test_dataset)}", leave=False)
            for epoch in epoch_progress:
                z_t_tmp = copy.deepcopy(z_t).repeat(args.q, 1)

                noise = torch.normal(mean=0.0, std=1.0, size=z_t_tmp.shape).to(args.device)
                noise = noise/torch.norm(noise, dim=1, keepdim=False)

                z_t_pos = z_t_tmp + args.mu * noise
                scores_pos = model_forward_api.eval(z_t_pos, batch_data['text'], scorer)
                f_scores_pos = scores_pos[0].to(args.device)

                z_t_neg = z_t_tmp - args.mu * noise
                scores_neg= model_forward_api.eval(z_t_neg, batch_data['text'], scorer, eval=True)
                f_scores_neg = scores_neg[0].to(args.device)



                g_t_hat = torch.mean(((f_scores_neg - f_scores_pos).reshape(-1, 1) * noise) / (2 * args.mu), dim=0)
                if args.alpha_w !=0:
                    total_g_t = torch.cat([window_dev, g_t_hat.unsqueeze(0)], dim=0)
                    total_g_t = total_g_t[1:, :]
                    g_t_finally = torch.matmul(total_g_t.T, window_alpha).squeeze(1)
                    g_t_finally = g_t_finally/torch.sum(window_alpha, dim=0, keepdim=False)
                    if args.use_confident==1:   
                        sign_match_mask = (g_t_finally * g_t_hat) > 0
                        g_t_finally = torch.where(sign_match_mask, g_t_finally, torch.zeros_like(g_t_finally))
                    if args.beta_w !=0:
                        g_squared = total_g_t ** 2
                        g_t_finally_B = torch.matmul(g_squared.T, window_beta).squeeze(1)
                        g_t_finally_B = g_t_finally_B/torch.sum(window_beta, dim=0, keepdim=False)
                        g_t_finally_B = torch.sqrt(g_t_finally_B)+torch.full_like(g_t_finally_B, args.eli)
                    if args.beta_w==0:
                        g_t_finally_B=1
                if args.alpha_w==0:
                    g_t_finally=g_t_hat
                    g_t_finally_B=1
                    total_g_t=0
                z_t = z_t - args.learning_rate * g_t_finally/g_t_finally_B
                window_dev = total_g_t
            
            optimized_prompt, score = model_forward_api.optimize_prompts(z_t.unsqueeze(0), batch_data['text'], scorer)
        else:
            optimized_prompt, score = model_forward_api.optimize_prompts(z_t.unsqueeze(0), batch_data['text'], scorer)
        Sum_score = Sum_score + score[0].item()
        if not os.path.exists(f'{save_dir}/{batch_idx}'):
            os.makedirs(f'{save_dir}/{batch_idx}')

        torch.save(model_forward_api.best_embedding, f'{save_dir}/{batch_idx}/final_prompt.pt')
        with open(f'{save_dir}/{batch_idx}/original_prompt.txt', "w", encoding="utf-8") as file:
            file.write(batch_data['text'][0])
        with open(f'{save_dir}/{batch_idx}/optimized_prompt.txt', "w", encoding="utf-8") as file:
            file.write(optimized_prompt[0] + "\n")
            file.write(f"Score: {score[0]}")
            file.write(f"Av_score: {Sum_score/(batch_idx+1)}")
        model_forward_api.reset()



if __name__ == '__main__':
    args = parse_args()
    set_all_seed(args.seed)
    run(args)
    print("Finished!")