import pytorch_lightning as pl
import clip
import numpy as np
import torch
import os

from torch import autocast, nn
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
from transformers import AutoProcessor, AutoModel


class AestheticMlp(pl.LightningModule):

  def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
    super().__init__()
    self.input_size = input_size
    self.xcol = xcol
    self.ycol = ycol
    self.layers = nn.Sequential(
      nn.Linear(self.input_size, 1024),
      #nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(1024, 128),
      #nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(128, 64),
      #nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(64, 16),
      #nn.ReLU(),
      nn.Linear(16, 1))

  def forward(self, x): return self.layers(x)


class PromptScorer:

  def __init__(self, sdmodel_name, cache_dir, device, num_images_per_prompt=3):
    
    # init scorer hparams
    self.lambda_aes = 0.05
    self.lambda_clip = 5.0
    self.num_images_per_prompt = num_images_per_prompt

    # init models
    self.sdmodel_name = sdmodel_name

    self.cache_dir = cache_dir
    self.device = device

    self.init_clip_model()
    self.init_aesthetic_model()
    self.init_diffusion_model()
    self.init_pickscore_model()

    self.eval_data_res = []
  
  def init_diffusion_model(self):
    diffusion_cache_dir = self.cache_dir + 'diffusion_model'
    if not os.path.exists(diffusion_cache_dir):
      os.makedirs(diffusion_cache_dir)
      print(f"Successfully create {diffusion_cache_dir} !")
    else:
      print(f"{diffusion_cache_dir} is existing !")
    
    device = self.device # device = f"cuda:{os.environ['LOCAL_RANK']}"
    # access_token = ""   # TODO Please provide the access token
    # dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained(self.sdmodel_name, subfolder="scheduler")
    # pipe = StableDiffusionPipeline.from_pretrained(self.sdmodel_name, use_auth_token=access_token, revision="fp16", 
    # torch_dtype=torch.float16, scheduler=dpm_scheduler)

    # dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained(self.sdmodel_name, subfolder="scheduler", cache_dir=diffusion_cache_dir)
    if self.sdmodel_name == 'sd-legacy/stable-diffusion-v1-5':
      pipe = StableDiffusionPipeline.from_pretrained(self.sdmodel_name, use_auth_token=True, #revision="fp16", 
      torch_dtype=torch.float16, cache_dir=diffusion_cache_dir)
    else:
      pipe = DiffusionPipeline.from_pretrained(self.sdmodel_name, torch_dtype=torch.bfloat16, cache_dir=diffusion_cache_dir, local_files_only=True)

    # Disable NSFW detect
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    self.diffusion_pipe = pipe

  def init_clip_model(self):
    clip_cache_dir = self.cache_dir + 'clip_model'
    if not os.path.exists(clip_cache_dir):
      os.makedirs(clip_cache_dir)
      print(f"Successfully create {clip_cache_dir} !")
    else:
      print(f"{clip_cache_dir} is existing !")
    #os.environ["CLIP_CACHE_PATH"] = clip_cache_dir

    device = self.device #device = f"cuda:{os.environ['LOCAL_RANK']}"
    self.clip_model, self.clip_preprocess = clip.load(clip_cache_dir+"/ViT-L-14.pt", device=device)

  def init_aesthetic_model(self):
    model = AestheticMlp(768)
    s = torch.load("./aesthetic/sac+logos+ava1-l14-linearMSE.pth")
    model.load_state_dict(s)
    device = self.device #device = f"cuda:{os.environ['LOCAL_RANK']}"
    model.to(device)
    model.eval()
    self.aes_model = model

  def init_pickscore_model(self):
    pickscore_cache_dir = self.cache_dir + 'pickscore_model'
    if not os.path.exists(pickscore_cache_dir):
      os.makedirs(pickscore_cache_dir)
      print(f"Successfully create {pickscore_cache_dir} !")
    else:
      print(f"{pickscore_cache_dir} is existing !")
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

    device = self.device

    self.pickscore_processor = AutoProcessor.from_pretrained(processor_name_or_path, cache_dir=pickscore_cache_dir)
    self.pickscore_model = AutoModel.from_pretrained(model_pretrained_name_or_path, cache_dir=pickscore_cache_dir).eval().to(device)
    
  def get_clip_features(self, pil_image, is_batched=False):
    if not is_batched:
      image = self.clip_preprocess(pil_image).unsqueeze(0)
    else:
      images = [self.clip_preprocess(i) for i in pil_image]
      image = torch.stack(images)
    device = self.device #device = f"cuda:{os.environ['LOCAL_RANK']}"
    image = image.to(device)
    with torch.no_grad():
      image_features = self.clip_model.encode_image(image)
    return image_features
  
  def get_clip_score(self, image_features, prompt):
    device = self.device #device = f"cuda:{os.environ['LOCAL_RANK']}"
    tokens = clip.tokenize([prompt], truncate=True).to(device)
    with torch.no_grad():
      text_features = self.clip_model.encode_text(tokens)
      image_features = image_features / image_features.norm(dim=1, keepdim=True)
      text_features = text_features / text_features.norm(dim=1, keepdim=True)
      logit_scale = self.clip_model.logit_scale.exp()
      # logit = logit_scale * image_features @ text_features.t()
      logit = image_features @ text_features.t()
      score = logit.item()
    return score
  
  def get_clip_score_batched(self, image_features, prompts):
    device = self.device #device = f"cuda:{os.environ['LOCAL_RANK']}"
    tokens = clip.tokenize(prompts, truncate=True).to(device)

    with torch.no_grad():
      if len(image_features) != len(prompts):
        assert len(image_features) % len(prompts) == 0
        tokens = tokens.unsqueeze(1).expand(-1, self.num_images_per_prompt, -1).reshape(-1, tokens.shape[-1])
      
      text_features = self.clip_model.encode_text(tokens)
      image_features = image_features / image_features.norm(dim=1, keepdim=True)
      text_features = text_features / text_features.norm(dim=1, keepdim=True)
      # logit_scale = self.clip_model.logit_scale.exp()
      logit = image_features @ text_features.t()
    scores = logit.diag().tolist()
    return scores

  def get_aesthetic_score(self, image_features, is_batched=False):
    features = image_features.cpu().detach().numpy()
    order = 2
    axis = -1
    l2 = np.atleast_1d(np.linalg.norm(features, order, axis))
    l2[l2 == 0] = 1
    im_emb_arr = features / np.expand_dims(l2, axis)
    #prediction = self.aes_model(torch.from_numpy(im_emb_arr).to(f"cuda:{os.environ['LOCAL_RANK']}").type(torch.cuda.FloatTensor))
    prediction = self.aes_model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
    if is_batched:
      return prediction[:, 0].tolist()
    else:
      return prediction.item()
    
  def get_pick_score(self, prompts, images):
    image_inputs = self.pickscore_processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(self.device)
    text_inputs = self.pickscore_processor(
        text=prompts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(self.device)

    with torch.no_grad():
        image_embs = self.pickscore_model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = self.pickscore_model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        scores = self.pickscore_model.logit_scale.exp() * torch.sum(text_embs.repeat_interleave(self.num_images_per_prompt,0) * image_embs, dim=1)
        #scores = self.pickscore_model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
    
    return scores.tolist()
  
  def gen_image(self, prompt):
    with autocast("cuda"):
      images = self.diffusion_pipe(prompt, guidance_scale=7.5, num_inference_steps=50).images
    return images[0]
  
  def gen_image_batched(self, prompts):
    images = []
    bsz = 1
    for i in range(0, len(prompts), bsz):
      pmpts = prompts[i: i + bsz]
      with autocast("cuda"):
        sub_images = self.diffusion_pipe(pmpts, num_images_per_prompt=self.num_images_per_prompt, num_inference_steps=20).images
        images.extend(sub_images)
    return images

  def get_score(self, prompt, plain_text):
    image = self.gen_image(prompt)
    image_features = self.get_clip_features(image)
    aes_score = self.get_aesthetic_score(image_features)
    clip_score = self.get_clip_score(image_features, plain_text)
    final_score = aes_score * self.lambda_aes + clip_score * self.lambda_clip
    return aes_score, clip_score, final_score
  
  def get_score_batched(self, prompts, plain_texts, plain_aes_score=None):
    images = self.gen_image_batched(prompts)
    image_features = self.get_clip_features(images, is_batched=True)
    aes_scores = self.get_aesthetic_score(image_features, is_batched=True)

    '''
    if plain_aes_score is None:
      images_plain = self.gen_image_batched(plain_texts)
      images_plain_features = self.get_clip_features(images_plain, is_batched=True)
      aes_scores_plain = self.get_aesthetic_score(images_plain_features, is_batched=True)

      images_plain = images_plain*len(prompts)
      images_plain_features = images_plain_features.repeat(len(prompts),1)
      aes_scores_plain = aes_scores_plain*len(prompts)
    else:
      aes_scores_plain = plain_aes_score.tolist()
    '''

    plain_texts = plain_texts*len(prompts)

    clip_scores = self.get_clip_score_batched(image_features, plain_texts)
    clip_scores = torch.Tensor(clip_scores)
    clip_scores = torch.maximum(clip_scores, torch.zeros_like(clip_scores))
    #clip_scores_plain = self.get_clip_score_batched(images_plain_features, plain_texts)
    #clip_scores_plain = torch.Tensor(clip_scores_plain)
    #clip_scores_plain = torch.maximum(clip_scores_plain, torch.zeros_like(clip_scores_plain))

    aes_scores = torch.Tensor(aes_scores)
    '''
    if len(aes_scores_plain) != len(aes_scores):
      aes_scores_plain = torch.Tensor(aes_scores_plain).unsqueeze(1).expand(-1, self.num_images_per_prompt).flatten()
    else:
      aes_scores_plain = torch.Tensor(aes_scores_plain)
    aes_scores_plain = aes_scores_plain.reshape(-1, self.num_images_per_prompt).mean(-1, keepdim=True).expand(-1, self.num_images_per_prompt).flatten()
    clip_scores_plain = clip_scores_plain.reshape(-1, self.num_images_per_prompt).mean(-1, keepdim=True).expand(-1, self.num_images_per_prompt).flatten()
    '''

    #final_scores = (aes_scores-aes_scores_plain) + torch.where(clip_scores>0.28, 0, 20*clip_scores-5.6)
    #final_scores = final_scores.reshape(-1, self.num_images_per_prompt).mean(1)
    # TODO: modify score calculation method

    #final_aes_scores = (aes_scores-aes_scores_plain).reshape(-1, self.num_images_per_prompt).mean(1)
    final_aes_scores = (aes_scores).reshape(-1, self.num_images_per_prompt).mean(1)
    #final_clip_scores = torch.where(clip_scores>0.28, 0, 20*clip_scores-5.6).reshape(-1, self.num_images_per_prompt).mean(1)
    #final_clip_scores = (clip_scores-clip_scores_plain).reshape(-1, self.num_images_per_prompt).mean(1)
    final_clip_scores = (clip_scores).reshape(-1, self.num_images_per_prompt).mean(1)*20


    # zhz add pickscore
    
    pick_scores = self.get_pick_score(plain_texts, images)
    #pick_scores_plain = self.get_pick_score(plain_texts, images_plain)
    pick_scores = torch.Tensor(pick_scores)
    #pick_scores_plain = torch.Tensor(pick_scores_plain)
    #pick_scores_plain = pick_scores_plain.reshape(-1, self.num_images_per_prompt).mean(-1, keepdim=True).expand(-1, self.num_images_per_prompt).flatten()
    #final_pick_scores = (pick_scores-pick_scores_plain).reshape(-1, self.num_images_per_prompt).mean(1)
    final_pick_scores = (pick_scores).reshape(-1, self.num_images_per_prompt).mean(1)*0.5
    

    return final_aes_scores, final_clip_scores, final_pick_scores #return final_scores