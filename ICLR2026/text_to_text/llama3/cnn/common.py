import torch
import numpy as np
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import random
from rouge_score import rouge_scorer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from zs_dm_cnn import generate_summary,calculate_rouge1_loss

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Set all the seeds to {seed} successfully!")

    
class LMForwardAPI:#
    def __init__(self, model_name=None, init_prompt=None, prompt_gen_template=None,
                 random_proj=None, intrinsic_dim=None, n_prompt_tokens=None, 
                 HF_cache_dir=None, demos=None, args=None):
        
        kwargs={
            'torch_dtype': torch.float16,
            'use_cache': True
            }
        self.ops_model = model_name

        ops_model_cache_dir = args.cache_dir + f'ops_model/{self.ops_model}'
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
        else:
            raise NotImplementedError

        self.init_token = init_prompt[0] #+ init_qa[0]
        if self.ops_model in ['wizardlm', 'vicuna', 'openchat']:
            self.embedding = self.model.get_input_embeddings().weight.clone()
            input_ids = self.tokenizer(init_prompt, return_tensors="pt").input_ids.cuda()
            self.init_prompt = self.embedding[input_ids]
            
        ################# setup n_prompts_token #################
        self.n_prompt_tokens = n_prompt_tokens
        self.hidden_size = self.init_prompt.shape[-1]
        print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))
        
        # self.init_prompt = self.init_prompt.reshape(self.n_prompt_tokens * self.hidden_size)
        # Create the template for Vicuna and WizardLM
        # self.count = 0
        self.linear = torch.nn.Linear(intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False).to(args.device)
        if self.ops_model == 'vicuna':
            self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The user gives a few examples of rephrasing and a sentence that needs to be rephrased. The assistant provides a rephrased sentence without additional content for user."
            self.role = ['USER:', 'ASSISTANT:']
        elif self.ops_model == 'wizardlm':
            self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives, helpful, detailed, and polite answers to the user's questions. Please provide a direct answer without additional content."
            self.role = ['USER:', 'ASSISTANT:']
        elif self.ops_model == 'alpaca':
            self.system_prompt= "Below is an instruction that describes a task. Write a response that appropriately completes the request. Please provide a direct answer without additional content."
            self.role = ["### Instruction:", "### Response:"]
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

        self.best_train_perf = 0.0
        self.best_dev_perf = -10.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0

        self.prompt_gen_template = prompt_gen_template

        self.demos = demos

        #self.gen_batch_size = args.gen_batch_size

    
    def reset(self):
        self.best_train_perf = 0.0
        self.best_dev_perf = -10.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0

    def remove_prefix(self, text):
        prefixes = ["Rephrased:", "Rephrase:", "Rephrasing:", "rephrased:", "rephrase:", "rephrasing:"]
        for prefix in prefixes:
            if text.startswith(prefix):
                return text[len(prefix):].strip() 
        return text
  
    def gen_prompt(self, data, prompt_embedding):

        #text_prompt = [self.init_token + self.prompt_gen_template.fill(self.demos, d) for d in data]
        text_prompt = [self.init_token + self.prompt_gen_template.fill({}, d) for d in data]
        input_text = [f"{self.system_prompt} USER:{p} ASSISTANT:" for p in text_prompt]

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
        
        prompt_input_embed = torch.cat((input_embed[:,:1,:], prompt_embedding, input_embed[:,1:,:]), dim=1)

        prompt_attention_mask = torch.cat((attention_mask, torch.ones(attention_mask.shape[0], self.n_prompt_tokens).to(attention_mask.device)), 1)
        outputs = self.model.generate(inputs_embeds=prompt_input_embed, attention_mask=prompt_attention_mask, max_new_tokens=512)
        instruction = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        instruction = [self.remove_prefix(ins) for ins in instruction]
        #print(instruction)

        return instruction

    
    def eval(self, prompt_embedding=None, batch_data=None, scorer=None,pipeline=None, ptype=None):

        q = prompt_embedding.shape[0]
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
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
        target = generate_summary(instructions,pipeline)
        if ptype == 0:
            scores = calculate_rouge1_loss(batch_data,target)
        if ptype == 1:

            scores = logic_matching_loss(batch_data,target)
        return scores
    

    def optimize_prompts(self, prompt_embedding, original_prompts, scorer):
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
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

        eval_score = scores[0].mean().item()

        if eval_score>self.best_dev_perf:
            self.best_dev_perf = eval_score
            self.best_prompt = instruction

        return self.best_prompt#linear
    

subsampled_data = (
    ['Astronaut rides horse.', 
     'A majestic sailing ship.',
     'Sunshine on iced mountain.',
     'Panda mad scientist mixing sparkling chemicals.'],
    ['Astronaut riding a horse, fantasy, intricate, elegant, highly detailed, artstation, concept art, smooth, sharp focus, illustration.',
     'A massive sailing ship, by Greg Rutkowski, highly detailed, stunning beautiful photography, unreal engine, 8K.',
     'Photo of sun rays coming from melting iced mountain, by greg rutkowski, 4 k, trending on artstation.',
     'Panda as a mad scientist, lab coat, mixing glowing and disinertchemicals, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration.']
)