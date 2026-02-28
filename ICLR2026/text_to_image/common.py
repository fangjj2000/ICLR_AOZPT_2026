import torch
import numpy as np
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
#from automatic_prompt_engineer import evaluate, config, template, data
import os
import random
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Set all the seeds to {seed} successfully!")

    
class LMForwardAPI:
    def __init__(self, model_name=None, init_prompt=None, prompt_gen_template=None,
                 random_proj=None, intrinsic_dim=None, n_prompt_tokens=None, 
                 HF_cache_dir=None, demos=None, args=None):
        # p = torch.ones(10)
        
        kwargs={
            'torch_dtype': torch.float16,
            'use_cache': True
            }
        self.ops_model = model_name
        # import pdb; pdb.set_trace()

        ops_model_cache_dir = args.cache_dir + f'ops_model'
        if not os.path.exists(ops_model_cache_dir):
            os.makedirs(ops_model_cache_dir)
            print(f"Successfully create {ops_model_cache_dir} !")
        else:
            print(f"{ops_model_cache_dir} is existing !")

        if self.ops_model in ["vicuna", "wizardlm", 'openchat']:
            self.model = AutoModelForCausalLM.from_pretrained(
                HF_cache_dir,
                low_cpu_mem_usage=True,
                device_map="auto",
                cache_dir=ops_model_cache_dir,
                **kwargs,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                                HF_cache_dir,
                                model_max_length=1024,
                                padding_side="left",
                                use_fast=False,
                                cache_dir=ops_model_cache_dir,
                            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        elif self.ops_model in ["promptist"]:
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist", device_map="auto", cache_dir=ops_model_cache_dir)
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=ops_model_cache_dir)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        else:
            raise NotImplementedError

        self.init_token = init_prompt[0] #+ init_qa[0]
        if self.ops_model in ['wizardlm', 'vicuna', 'openchat', 'promptist']:
            self.embedding = self.model.get_input_embeddings().weight.clone().cuda()
            input_ids = self.tokenizer(init_prompt, return_tensors="pt").input_ids.cuda()
            self.init_prompt = self.embedding[input_ids]
            
        ################# setup n_prompts_token #################
        self.n_prompt_tokens = n_prompt_tokens
        self.hidden_size = self.init_prompt.shape[-1]
        # print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))
        
        # self.init_prompt = self.init_prompt.reshape(self.n_prompt_tokens * self.hidden_size)
        # Create the template for Vicuna and WizardLM
        # self.count = 0
        self.linear = torch.nn.Linear(intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False).to(args.device)
        if self.ops_model == 'vicuna':
            self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The user gives a few examples of rephrasing and a sentence that needs to be rephrased. The assistant provides a rephrased sentence without additional content for user."
            #self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The user gives a few examples and a simple description. The assistant provides a drawing prompt from the description without additional content for user."
            #self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The user gives a simple description of an image. The assistant provides a drawing prompt from the description without additional content for user."
            self.role = ['USER:', 'ASSISTANT:']
        elif self.ops_model == 'wizardlm':
            self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            self.role = ['USER:', 'ASSISTANT:']
        elif self.ops_model == 'alpaca':
            self.system_prompt= "Below is an instruction that describes a task. Write a response that appropriately completes the request. Please provide a direct answer without additional content."
            self.role = ["### Instruction:", "### Response:"]
        elif self.ops_model == 'promptist':
            self.system_prompt = ""
            self.role = ['USER:', 'ASSISTANT:']
        elif self.ops_model == 'openchat':
            self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
            self.role = ['user:', 'content:']
        else:
            NotImplementedError
            

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

        ## eval preparation
        ##self.conf = config.update_config(conf, base_conf)
        ##self.eval_data = eval_data
        ##self.eval_template = template.EvalTemplate("Instruction: [PROMPT]\n\nInput: [INPUT]\n Output: [OUTPUT]")
        ##self.demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")

        # Temporarily remove the API model "LLaMA-33B" and "Flan-T5 13B" 
        # if args.api_model in ['llama', 'flan-t5']:
        #     self.api_model = exec_evaluator(args.api_model, self.conf)
        # else:
        ##self.api_model = args.api_model

        # if few_shot_data is None:
        #     self.few_shot_data = prompt_gen_data
        
        self.best_train_perf = 0.0
        self.best_dev_perf = -10.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.best_embedding = None
        self.num_call = 0

        self.prompt_gen_template = prompt_gen_template

        self.demos = demos

        self.zero_shot = args.zero_shot

        #self.gen_batch_size = args.gen_batch_size

    
    def reset(self):
        self.best_train_perf = 0.0
        self.best_dev_perf = -10.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.best_embedding = None
        self.num_call = 0

    def remove_prefix(self, text):
        prefixes = ["Rephrased:", "Rephrase:", "Rephrasing:", "rephrased:", "rephrase:", "rephrasing:", "prompt:", "Prompt:"]
        for prefix in prefixes:
            start_index = text.find(prefix)
            if start_index != -1:
                return text[start_index + len(prefix):].strip()
        return text

    
    def gen_prompt(self, data, prompt_embedding):

        if self.ops_model == 'promptist':
            text_prompt = [d for d in data]
            input_text = [f"{p} Rephrase:" for p in text_prompt]
        else:
            text_prompt = [self.init_token + self.prompt_gen_template.fill(self.demos, d) for d in data]
            input_text = [f"{self.system_prompt}\nUSER: {p}\nASSISTANT:" for p in text_prompt]

        input_token = self.tokenizer(input_text, return_tensors="pt", padding=True)
        input_ids = input_token.input_ids.cuda()
        attention_mask = input_token.attention_mask.cuda()

        input_embed = self.embedding[input_ids]
        prompt_embedding = prompt_embedding.to(device=input_embed.device, dtype=input_embed.dtype)

        if input_embed.shape[0] == 1 and input_embed.shape[0] != prompt_embedding.shape[0]:
            input_embed = input_embed.repeat(prompt_embedding.shape[0],1,1)
            attention_mask = attention_mask.repeat(prompt_embedding.shape[0],1)
        elif input_embed.shape[0] != 1:
            raise ValueError(
                f'[input_embed] Only support 1, got `{input_embed.shape[0]}` instead.'
            )
        
        if self.zero_shot:
            prompt_input_embed = input_embed
            prompt_attention_mask = attention_mask
        else:
            #prompt_input_embed = torch.cat((input_embed[:,:1,:], prompt_embedding, input_embed[:,1:,:]), dim=1)
            prompt_input_embed = torch.cat((prompt_embedding, input_embed), dim=1)
            prompt_attention_mask = torch.cat((attention_mask, torch.ones(attention_mask.shape[0], self.n_prompt_tokens).to(attention_mask.device)), 1)

        if self.ops_model == 'promptist':
            eos_id = self.tokenizer.eos_token_id
            outputs = self.model.generate(
                inputs_embeds=prompt_input_embed,
                attention_mask=prompt_attention_mask,
                do_sample=False, 
                max_new_tokens=75, 
                num_beams=8, 
                num_return_sequences=1,
                eos_token_id=eos_id, 
                pad_token_id=eos_id, 
                length_penalty=-1.0
            )
        else:
            outputs = self.model.generate(inputs_embeds=prompt_input_embed, attention_mask=prompt_attention_mask, max_new_tokens=256)
        instruction = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        instruction = [self.remove_prefix(ins) for ins in instruction]
        #print(instruction)

        return instruction

    
    def eval(self, prompt_embedding=None, batch_data=None, scorer=None, eval=False, weights=None, baseline_score=None):
        q = prompt_embedding.shape[0]
        tmp_prompt = copy.deepcopy(prompt_embedding)
        if isinstance(prompt_embedding, torch.Tensor): 
            prompt_embedding = prompt_embedding.type(torch.float32)
            prompt_embedding = self.linear(prompt_embedding)  # Az
            prompt_embedding = prompt_embedding.reshape(q, self.n_prompt_tokens, -1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [torch.Tensor], got `{type(prompt_embedding)}` instead.'
            )
        
        instructions = self.gen_prompt(batch_data, prompt_embedding)

        scores = scorer.get_score_batched(instructions, batch_data)
        return scores
    

    def optimize_prompts(self, prompt_embedding, original_prompts, scorer, weights=None, baseline_score=None):
        tmp_prompt = copy.deepcopy(prompt_embedding)
        if isinstance(prompt_embedding, torch.Tensor): 
            prompt_embedding = prompt_embedding.type(torch.float32)
            prompt_embedding = self.linear(prompt_embedding)  # Az
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [torch.Tensor], got `{type(prompt_embedding)}` instead.'
            )
        
        instruction = self.gen_prompt(original_prompts, prompt_embedding)
        scores = scorer.get_score_batched(instruction, original_prompts)

        return instruction, scores


    def cal_baseline(self, original_prompts, scorer):
        scorer.num_images_per_prompt=10
        scores = scorer.get_score_batched(original_prompts, original_prompts)
        scorer.num_images_per_prompt=3
        return scores


    def case_study_eval(self, prompt_embedding=None, batch_data=None, scorer=None):
        q = prompt_embedding.shape[0]
        if isinstance(prompt_embedding, torch.Tensor): 
            prompt_embedding = prompt_embedding.type(torch.float32)
            prompt_embedding = self.linear(prompt_embedding)  # Az
            prompt_embedding = prompt_embedding.reshape(q, self.n_prompt_tokens, -1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [torch.Tensor], got `{type(prompt_embedding)}` instead.'
            )
        
        instructions = self.gen_prompt(batch_data, prompt_embedding)

        images = scorer.gen_image_batched(instructions)

        return images, instructions


def split_text_by_words(text, words_per_line):
    words = text.split()
    lines = [' '.join(words[i:i + words_per_line]) for i in range(0, len(words), words_per_line)]
    return '\n'.join(lines)


def display(generated_images, instructions, maximum_display_rows = 2):
    nrows = len(generated_images) // maximum_display_rows + 1
    if len(generated_images) % maximum_display_rows == 0:
        nrows -= 1
    ncols = min([maximum_display_rows, len(generated_images)])
    fig, ax = plt.subplots(nrows, ncols, figsize=(12*nrows,6*ncols),dpi=500, constrained_layout=True)
    ind = range(len(generated_images))
    if nrows > 1:
        for i in range(nrows):
            for j in range(ncols):

                nq = i * ncols + j
                ax[i][j].axis('off')
                if nq < len(ind):
                    t_ind = ind[nq]
                    ax[i][j].imshow(generated_images[t_ind])
                    fig_id = "ID:{}, {}".format(nq+1, split_text_by_words(instructions[t_ind], 8))
                    ax[i][j].set_title(fig_id)
                else:
                    fig.delaxes(ax[i][j])
    else:
        for nq in range(ncols):
            ax[nq].axis('off')
            if nq < len(ind):
                t_ind = ind[nq]
                ax[nq].imshow(generated_images[t_ind])
                fig_id = "ID:{}, {}".format(nq+1, split_text_by_words(instructions[t_ind], 8))
                ax[nq].set_title(fig_id)
    plt.show()
    return 0