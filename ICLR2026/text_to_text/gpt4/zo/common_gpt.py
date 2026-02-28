import os
import numpy as np
import torch
import datasets
from datasets import Metric, MetricInfo
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import time
from openai import OpenAI
import re
from tqdm import tqdm

DOMAIN_DATASET = ['CI', 'SE', 'RCT', 'HP']

task_to_keys = {
    "cola": ("sentence", None),
    "book": ("sentence", None),
    "elec": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "snli": ("premise", "hypothesis"),
}

LABEL2ID_CONFIG = {
    "sst2": {" terrible": 0, " great": 1},
    "qqp": {" no": 0, " yes": 1},
    "mrpc": {" no": 0, " yes": 1},
    "cola": {" no": 0, " yes": 1},
    "wnli": {" no": 0, " yes": 1},
    "qnli": {" yes": 0, " no": 1},
    "rte": {" yes": 0, " no": 1},
    # "book": {" great": 0, " terrible": 1},
    "elec": {" great": 0, " terrible": 1},
    "book": {" positive": 0, " negative": 1},
    #"elec": {" positive": 0, " negative": 1},
    #"book": {" yes": 0, " no": 1},
    #"elec": {" yes": 0, " no": 1},
    "imdb": {" terrible": 0, " great": 1},
    "cr": {" terrible": 0, " great": 1},
    "mr": {" terrible": 0, " great": 1},
    "HP": {' unhelpful': 0, ' helpful': 1}, # review helpfulness
    "mpqa": {" terrible": 0, " great": 1},
    "mnli": {" no": 0, " maybe": 1, " yes": 2},
    "snli": {" yes": 0, " maybe": 1, " no": 2},
    "CI": {' background': 0, ' comparison': 1, ' extension': 2, ' future': 3, ' motivation': 4, ' use': 5},
    "SE": {' comparison': 0, ' conjunction': 1, ' evaluation': 2, ' feature': 3, ' hyponym': 4, ' part': 5, ' function': 6},
    "RCT": {' background': 0, ' conclusion': 1, ' method': 2, ' objective': 3, ' result': 4} ,
}

LABEL_CONVERT = {
    "sst2": {0: ' terrible', 1: ' great'},
    "qqp": {0: ' no', 1: ' yes'},
    'mrpc': {0: ' no', 1: ' yes'},
    'cola': {0: ' no', 1: ' yes'},
    'wnli': {0:  ' no', 1: ' yes'},
    'qnli': {0: ' yes', 1: ' no'},
    'rte': {0: ' yes', 1: ' no'},
    "mnli": {0: ' no', 1: ' maybe', 2: ' yes'},
    # "book":{0: ' great', 1: ' terrible'},
    "elec":{0: ' great', 1: ' terrible'},
    "book":{0: ' positive', 1: ' negative'},
    #"elec":{0: ' positive', 1: ' negative'},
    #"book":{0: ' yes', 1: ' no'},
    #"elec":{0: ' yes', 1: ' no'},
    "snli": {0: ' yes', 1: ' maybe', 2: ' no'},
    'CI': {'Background': ' background', 'CompareOrContrast': ' comparison', 'Extends': ' extension', 'Future': ' future', 'Motivation': ' motivation', 'Uses': ' use'},
    'SE': {'COMPARE': ' comparison', 'CONJUNCTION': ' conjunction', 'EVALUATE-FOR': ' evaluation', 'FEATURE-OF': ' feature', 'HYPONYM-OF': ' hyponym', 'PART-OF': ' part', 'USED-FOR': ' function'},
    'RCT': {'BACKGROUND': ' background', 'CONCLUSIONS': ' conclusion', 'METHODS': ' method', 'OBJECTIVE': ' objective', 'RESULTS': ' result'},
    'HP': {False: ' unhelpful', True: ' helpful'},
}

TEMPLATE_CONFIG = {
    "mnli": " entailment?",
    "qqp": " equivalent?",
    # "sst2": " What is the sentiment?",
    "mrpc": " equivalent?",
    "cola": " correct?",
    "book": " It was ",
    "elec": " It was ",
    "wnli": " What is the relation?",
    "qnli": " entailment?",
    "rte": " entailment?",
    "CI": " What is the intent?",
    "SE": " What is the relation?",
    "RCT": " What is the role?",
    "HP": " Helpful?",
    "sst2": " It was ",
    "imdb": " It was ",
    "cr": " It was ",
    "snli": " entailment?",
}#task

def solve_v_total_exact(prompt_emb):
    k = 1
    a, b = -3, 0

    b = prompt_emb.max()
    def f(v):
        s = (prompt_emb - v).clamp(0, 1).sum()
        return s - k
    itr = 0

    v = 0
    while (1):
        itr += 1
        v = (a + b) / 2
        obj = f(v)
        if abs(obj) < 1e-3 or itr > 20:
            break
        if obj < 0:
            b = v
        else:
            a = v
    return v, itr

def constrainScoreByWholeExact(prompt_embeds):
    for i in range(len(prompt_embeds)):
        v, itr = solve_v_total_exact(prompt_embeds[i])
        prompt_embeds[i].sub_(v).clamp_(0, 1)
#gpt4_
def pmi(args) -> list:
    if args.use_ngram:
        prefix = {"gpt2": "gpt2-xl_", "gpt2-xl": "gpt2-xl_", "roberta-large": "", "llama3": "llama3_", "gpt-3.5-turbo": "gpt3-turbo_", "gpt-4": "gpt4_cnn_dailymail"}
        flag_name = args.file_name if args.file_name else args.task_name
        result=[]
        with open(f"/hy-tmp/wangql/online_learning/fjj_icml_BDPL/bdpl/pmi/{args.model_name_or_path}/{prefix[args.model_name_or_path]}" + flag_name.lower() + ".txt",'r') as f:
            for line in f:
                result = result + (list(line.strip('\n').split(',')))

        unique = []
        [unique.append(i) for i in result if not i in unique]
        if type(unique[0]) == int:
            ngram_index_list = list(map(int, unique))
        else:
            ngram_index_list = list(unique)
        return ngram_index_list
    return []

def counter(func):
    def wrapper(*args, **kwargs):
        wrapper.count = wrapper.count + 1
        res = func(*args, **kwargs)
        if wrapper.count % 100 == 0:
            # print ("{0} has been used: {1}x".format(func.__name__, wrapper.count))
            pass
        return res
    wrapper.count = 0
    return wrapper

class ApiCallLimitError(Exception):
    pass

def evaluate(args, model, eval_dataloader, metric, accelerator, epoch, api_count, prompts_probs=None, prompt_length=None, tokenizer=None, folder=None):
    prompts_discrete_indices = prompts_probs.argmax(1)

    if args.use_ngram:
        prompts_discrete_indices_ngram_list = []
        indices_list = prompts_discrete_indices.int().tolist()
        for idx in indices_list:
            prompts_discrete_indices_ngram_list.append(args.ngram_list[idx])
        prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)
    
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if args.trial and step >= 100:
                break
            bsz = len(batch['input_ids'])

            if args.use_ngram:
                # batch['input_ids'] = torch.cat([tokenizer.bos_token_id + torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
                batch['input_ids'] = torch.cat([prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
            else:
                # batch['input_ids'] = torch.cat([tokenizer.bos_token_id + torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
                batch['input_ids'] = torch.cat([prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
            
            # batch["attention_mask"] = torch.cat([torch.ones(bsz, 1).to(args.device), torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]], dim=1)
            batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]], dim=1)

            # mask_pos=np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
            # mask_pos = torch.tensor(mask_pos[-1])
            label_to_id = model.config.label2id 
            
            max_sentence_index = max(torch.sum(batch['input_ids'] != tokenizer.eos_token_id, dim=1))
            sequence_output = model(input_ids=batch['input_ids'][:, :max_sentence_index], attention_mask=batch["attention_mask"][:, :max_sentence_index])
            logits_out = sequence_output['logits']
            logits = logits_out[torch.arange(logits_out.size(0)), torch.sum(batch["attention_mask"], dim=1, dtype=int) - 1]
            # logits = sequence_output['logits'][:, -1]
            # print(logits)
            # last_hidden_state = sequence_output[0].squeeze()
            # logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

            label = batch["labels"].to(args.device)
            label_keys = list(label_to_id.keys())
            label_map = {}
            for target in label_keys:
                label_map[tokenizer.encode(target, add_special_tokens=False)[0]] = label_to_id[target]
            converted_target = label.clone()
            for key, val in label_map.items():
                converted_target[label == key] = val
            sort_label_map = sorted(label_map.items(), key=lambda item: item[1])
            interest_index = [item[0] for item in sort_label_map]
            logits = logits[:, interest_index]
            #predictions = logits.argmax(dim=-1)
            
            # predictions = (torch.exp(logits)/torch.sum(torch.exp(logits), dim=1).unsqueeze(1))[:, 1]
            
            if args.task_name in {'mnli', 'snli'}:
                predictions_pred = torch.nn.functional.softmax(logits, dim=1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions_pred.cpu().numpy()),
                    references=accelerator.gather(converted_target.cpu().numpy()),
                )
            else:
                probs = torch.nn.functional.softmax(logits, dim=1)
                probs = probs[:, 1]
                metric.add_batch(
                    predictions=accelerator.gather(probs),
                    references=accelerator.gather(converted_target),
                )

    if args.file_name in DOMAIN_DATASET:
        eval_metric = metric.compute(average='macro')
    else:
        eval_metric = metric.compute()

    # print("** eval **")
    # print(f"epoch {epoch + 1}: {eval_metric}")
    if args.balance == True:
        eval_result = eval_metric['acc']
    else:
        eval_result = eval_metric['acc']#eval_result = eval_metric['auc']
    write_results(args, folder, epoch, api_count, None, None, eval_metric, metric_state='eval')
    return eval_result

def test(args, model, test_dataloader, metric, accelerator, epoch, api_count, best_epoch, best_api_count, prompts_probs=None, prompt_length=None, tokenizer=None, test_dataloader_mm=None, folder=None, test_metric=None):
    if test_metric is not None:
        write_results(args, folder, epoch, api_count, best_epoch, best_api_count, test_metric, metric_state='test')
        return test_metric
    
    prompts_discrete_indices = prompts_probs.argmax(1)
    if args.use_ngram:
        prompts_discrete_indices_ngram_list = []
        indices_list = prompts_discrete_indices.int().tolist()
        for idx in indices_list:
            prompts_discrete_indices_ngram_list.append(args.ngram_list[idx])
        prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)

    test_logits = []
    test_labels = []

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            if args.trial and step >= 100:
                break
            bsz = len(batch['input_ids'])
            
            if args.use_ngram:
                # batch['input_ids'] = torch.cat([tokenizer.bos_token_id + torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
                batch['input_ids'] = torch.cat([prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
            else:
                # batch['input_ids'] = torch.cat([tokenizer.bos_token_id + torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
                batch['input_ids'] = torch.cat([prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids']], dim=1)
            batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]], dim=1)

            # mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
            # mask_pos = torch.tensor(mask_pos[-1])
            label_to_id = model.config.label2id 
            
            max_sentence_index = max(torch.sum(batch['input_ids'] != tokenizer.eos_token_id, dim=1))
            sequence_output = model(input_ids=batch['input_ids'][:, :max_sentence_index], attention_mask=batch["attention_mask"][:, :max_sentence_index])
            logits_out = sequence_output['logits']
            logits = logits_out[torch.arange(logits_out.size(0)), torch.sum(batch["attention_mask"], dim=1, dtype=int) - 1]
            # logits = sequence_output['logits'][:, -1]

            label = batch["labels"].to(args.device)
            label_keys = list(label_to_id.keys())
            label_map = {}
            for target in label_keys:
                label_map[tokenizer.encode(target, add_special_tokens=False)[0]]  = label_to_id[target]
            converted_target = label.clone()
            for key, val in label_map.items():
                converted_target[label == key] = val
            sort_label_map = sorted(label_map.items(), key=lambda item: item[1])
            interest_index = [item[0] for item in sort_label_map]
            logits = logits[:, interest_index]
            #predictions = logits.argmax(dim=-1)
            
            # predictions = (torch.exp(logits)/torch.sum(torch.exp(logits), dim=1).unsqueeze(1))[:, 1]
            
            if args.task_name in {'mnli', 'snli'}:
                predictions_pred = torch.nn.functional.softmax(logits, dim=1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions_pred.cpu().numpy()),
                    references=accelerator.gather(converted_target.cpu().numpy()),
                )
            else:
                probs = torch.nn.functional.softmax(logits, dim=1)
                probs = probs[:, 1]
                metric.add_batch(
                    predictions=accelerator.gather(probs),
                    references=accelerator.gather(converted_target),
                )
            
    if args.file_name in DOMAIN_DATASET:
        test_metric = metric.compute(average='macro')
    else:
        test_metric = metric.compute()

    # print("** test **")
    # print(f"current epoch {epoch + 1}, best epoch {best_epoch + 1}, api_count {api_count}: {test_metric}")
    if args.use_wandb:
        for key in test_metric.keys():
            eval_key = 'Black_test_' + key
            wandb.log({eval_key: test_metric[key]})
    
    write_results(args, folder, epoch, api_count, best_epoch, best_api_count, test_metric, metric_state='test', test_logits=test_logits, test_labels=test_labels)
    return test_metric

def write_results(args, folder, epoch, api_count, best_epoch, best_api_count, metric, metric_state, test_logits=None, test_labels=None):
    assert metric_state in ['eval', 'test']
    file_path = f'auc/{folder}/' + f'{args.api_limit}_{args.prompt_length}_{args.prompt_search_space}_{args.k_shot}_{args.prompt_learning_rate}'
    file_path = file_path if args.param_learning_rate is None else file_path + f'_{args.param_learning_rate}'
    file_path = file_path + f'_{args.loss_type}/seed_{args.seed}'
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    result_path = os.path.join(file_path, f'{args.task_name}_result.txt')
    with open(result_path, 'a') as f:
        if metric_state == 'eval':
            f.write(f"current_epoch {epoch + 1}, total_api_count {api_count}, {metric_state}_metric_results {metric} \n")
        else:
            f.write(f"current_epoch {epoch + 1}, total_api_count {api_count}, best_eval_epoch {best_epoch + 1}, best_api_count {best_api_count}, {metric_state}_metric_results {metric} \n")
        if epoch + 1 == args.num_train_epochs and metric_state == 'test':
            f.write(f"finished\n")
    
    if metric_state == 'test' and test_logits != None and test_labels != None:
        logits_path = os.path.join(file_path, f'{args.task_name}_logits.pt')
        torch.save(test_logits, logits_path)

        labels_path = os.path.join(file_path, f'{args.task_name}_labels.pt')
        torch.save(test_labels, labels_path)

def simple_accuracy(preds, labels):
    return float((np.array(preds) == np.array(labels)).mean())

def get_batch_indices(start, end, indices):
    batch_indices = np.where((indices >= start) & (indices < end))[0]
    if len(batch_indices) > 0:
        batch_indices = indices[batch_indices] - start
    return batch_indices


class CompleteGPT():
    def __init__(self):
        self.client = OpenAI(
            api_key = "",
            base_url = ""
        )

    def complete_gpt3(self, prompt, max_tokens, model_name, n=1, top_logprob=1):
        received = False
        while not received:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=prompt,
                    max_tokens=max_tokens,  
                    temperature=0.7,
                    n=n,                      
                )
                received = True
            except Exception as error:
                print("An error occurred in complete_gpt3:", error)
        content = response.choices[0].message.content
        if isinstance(content, list):
            text = "".join(
                part["text"] if isinstance(part, dict) else getattr(part, "text", "")
                for part in content
                if (isinstance(part, dict) and part.get("type") == "text") or hasattr(part, "text"))
        else:
            text = content 
        return text
    @counter
    def train_api_request(self, prompt, max_tokens, model_name, n=1, top_logprob=1):
        response=self.complete_gpt3(prompt, max_tokens, model_name, n=n, top_logprob=top_logprob)
        return response
    
    def eval_api_request(self, prompt, max_tokens, model_name, n=1, top_logprob=1):
        response=self.complete_gpt3(prompt, max_tokens, model_name, n=n, top_logprob=top_logprob)
        return response

    class ApiCallLimitError(Exception):
        pass

    def get_label_prob(self, response, chat_obj, label_keys, args, prob_if_label_not_found=0.1, tokenizer=None):
        gournd_truth = [token.replace("Ġ", " ") for token in tokenizer.tokenize(label_keys)]
        gournd_truth = [token.replace("Ċ", "\n") for token in gournd_truth]
        #labels_prob = torch.zeros(len(gournd_truth))
        #print(chat_obj)
        #print(response.choices[0].message.content)
        text_length = min(len(gournd_truth), len(response.choices[0].logprobs.content))
        labels_prob = torch.zeros(text_length)
        for label_index in range(text_length):  
            label = gournd_truth[label_index]
            found_the_label = False
            #print(f"finding {label}.", end=" ")
            for j in range(len(response.choices[0].logprobs.content[0].top_logprobs)): # for i in range(len(response.choices[0].logprobs.content)):  J first because, we want the top prob first. 
                if label.lower() == response.choices[0].logprobs.content[label_index].top_logprobs[j].token.lower():
                    prob = response.choices[0].logprobs.content[label_index].top_logprobs[j].logprob
                    labels_prob[label_index] = prob
                    found_the_label = True
                if found_the_label: break
            # be careful about the indent. 
            if not found_the_label:
                #print(f"xxx<{label}>xxx", end=" ")
                ##labels_prob[label_index] = -np.log(prob_if_label_not_found) # small probl
                labels_prob[label_index] = response.choices[0].logprobs.content[label_index].top_logprobs[-1].logprob
  
        """
        if label in response.choices[0].logprobs.content[0].token:
            label_prob = np.exp(response.choices[0].logprobs.content[0].logprob)
            return label_prob
        else:
            missing_response = self.train_api_request(chat_obj, l=100, model_name=args.model_name_or_path, n=1, top_logprob=1)
            print("the missing label is : ", label)
            for i in range(20):
                if label == missing_response.choices[0].logprobs.content[0].top_logprobs[i].token:
                    label_prob = np.exp(response.logprobs.content[0].logprob)
                    return label_prob
        """
        #print("\n", labels_prob)
        return labels_prob # a small label. 

def get_answer(response):
    match = re.search(r'####\s*(-?\d+(\.\d+)?)$', response)  
    if match:
        return float(match.group(1))  
    else:
        return None

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

def evaluateGPT3(args, eval_batches, loss, accelerator, epoch, results, prompts_probs=None, prompt_length=None, tokenizer=None):
        
    if prompts_probs is not None:
        prompts_discrete_indices = prompts_probs.argmax(1)

        if args.use_ngram:
            prompts_discrete_ngram_list = []
            indices_list = prompts_discrete_indices.int().tolist()
            for idx in indices_list:
                prompts_discrete_ngram_list.append(args.ngram_list[idx])
            prompts_discrete = ' '.join(prompts_discrete_ngram_list)

        else: 
            indices_list = prompts_discrete_indices.int().tolist()
            prompts_discrete = tokenizer.decode(indices_list, clean_up_tokenization_spaces=False)

    complete_GPT = CompleteGPT()
    pred_list = []
    for step in range(len(eval_batches['sentence'])):
        if args.trial and step >= args.trial_step:
            break

        labels = eval_batches["labels"][step]
        
        batch = []                                                          
        label_probs = []
        for i in range(len(eval_batches['sentence'][step])): #  change to single one each. 
            chat_obj = [
                #{ "role": "system", "content": "Place the result number after ' #### ' at the end of the answer, for example ' #### 0'."},
                { "role":'user', "content" : prompts_discrete + " Please give a summary of the following news." + "\nNews: " + eval_batches['sentence'][step][i] + "\nSummary:" }
            ]
            label = eval_batches['labels'][step][i]

            ground_truth = label
            # 
            response = complete_GPT.eval_api_request(chat_obj, max_tokens=100, model_name=args.model_name_or_path, n=1, top_logprob=5)
            #labels_prob = complete_GPT.get_label_prob(response, chat_obj, label, args, tokenizer=tokenizer)
            batch.append(chat_obj)
            #print(labels_prob)   
            # label_probs.append(-sum(labels_prob)) # if the prompt cannto get, it will be -10, meaning that it is very small. 

            pred = response.choices[0].message.content
            rouge = scorer.score(ground_truth, pred)["rouge1"].fmeasure
            pred_list.append(rouge)
            # if ground_truth != None and pred != None and ground_truth == pred:
            #     pred_list.append(1)
            # else:
            #     pred_list.append(0)
        
        # end. 
    eval_acc = np.mean(pred_list)
    print("** eval **")
    print(f"epoch {epoch}: {eval_acc}")

    eval_result = eval_acc
    results.append(eval_result)
    
    return eval_result


def testGPT3(args, test_batches, loss, accelerator, epoch, api_count, best_epoch, best_api_count, results, prompts_probs=None, prompt_length=None, tokenizer=None, test_metric=None):
    if test_metric is not None:
        # write_results(args, folder, epoch, api_count, best_epoch, best_api_count, test_metric, metric_state='test')
        return test_metric
    
    prompts_discrete = None
    if prompts_probs is not None:
        prompts_discrete_indices = prompts_probs.argmax(1)

        if args.use_ngram:
            prompts_discrete_ngram_list = []
            indices_list = prompts_discrete_indices.int().tolist()
            for idx in indices_list:
                prompts_discrete_ngram_list.append(args.ngram_list[idx])
            prompts_discrete = ' '.join(prompts_discrete_ngram_list)
        else: 
            indices_list = prompts_discrete_indices.int().tolist()
            prompts_discrete = tokenizer.decode(indices_list, clean_up_tokenization_spaces=False)

    complete_GPT = CompleteGPT()
    pred_list = []
    if prompts_discrete is not None:
        print('best prompt: ', prompts_discrete)
    for step in tqdm(range(len(test_batches['sentence']))):
        if args.trial and step >= args.trial_step:
            break
        labels = test_batches["labels"][step]

        # start
        
        batch = []                                                         
        label_probs = []
        for i in range(len(test_batches['sentence'][step])): #  change to single one each. 
            if prompts_discrete is not None:
                chat_obj = [
                    #{ "role": "system", "content": "Place the result number after ' #### ' at the end of the answer, for example ' #### 0'."},
                    { "role":'user', "content" : prompts_discrete + " Please give a summary of the following news." + "\nNews: " + test_batches['sentence'][step][i] + "\nSummary:" }
                ]
            else:
                chat_obj = [
                    #{ "role": "system", "content": "Place the result number after ' #### ' at the end of the answer, for example ' #### 0'."},
                    { "role":'user', "content" : "Please give a summary of the following news." + "\nNews: " + test_batches['sentence'][step][i] + "\nSummary:" }
                ]
            label = test_batches['labels'][step][i]

            ground_truth = label

            response = complete_GPT.eval_api_request(chat_obj, max_tokens=100, model_name=args.model_name_or_path, n=1, top_logprob=5)
            #labels_prob = complete_GPT.get_label_prob(response, chat_obj, label, args, tokenizer=tokenizer)
            batch.append(chat_obj)
            #print(labels_prob)   
            #label_probs.append(-sum(labels_prob)) # if the prompt cannto get, it will be -10, meaning that it is very small. 

            pred = response.choices[0].message.content
            print('ground_truth: ', label)
            print('response: ', response.choices[0].message.content)
            rouge = scorer.score(ground_truth, pred)["rouge1"].fmeasure
            pred_list.append(rouge)
            # if ground_truth != None and pred != None and ground_truth == pred:
            #     pred_list.append(1)
            #     print('yes')
            # else:
            #     pred_list.append(0)
            #     print('no')
        
        #label_probs = self.complete_GPT.get_regular_label_probs(responses, batch, label_keys, args, if_null = True)

    test_acc = np.mean(pred_list)
    
    print("** test **")
    print(f"epoch {epoch}: {test_acc}")
    
    test_result = test_acc
    results.append(test_result)
    
    # write_results(args, folder, epoch, api_count, best_epoch, best_api_count, test_metric, metric_state='test')
    return test_result