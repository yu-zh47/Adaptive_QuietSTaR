import torch
torch.backends.cuda.matmul.allow_tf32 = True
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from datasets import load_dataset
import os
import time
import re
from tqdm import tqdm
from collections import Counter
import distributed as dist
import itertools
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--batch_idx", type=int, default=0)
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--device_batch_size", type=int, default=8) # changed
parser.add_argument("--max_idx", type=int, default=128)
### n_votes, previously used 8
parser.add_argument("--n_votes", type=int, default=8)
###
parser.add_argument("--temp", type=float, default=0.9)
parser.add_argument("--start_final_answer_idx", type=int, default=384)
parser.add_argument("--answer_length", type=int, default=12)
parser.add_argument("--root_prefix", type=str, default="YOUR_ROOT_HERE")
parser.add_argument("--checkpoint", type=str, default="ezelikman/quietstar-8-ahead")
parser.add_argument("--final_answer_text", type=str, default="\nTherefore, the answer (arabic numerals) is")
parser.add_argument("--zero_shot_cot_prompt", type=str, default="\nA: Let's think step by step.")
### change for thought token length
parser.add_argument("--n_ahead", type=int, default=14)
###
args = parser.parse_args()


def model_init(params):
    if params is None:
        params = {}
    else:
        params = params.params
    n_ahead = params.get("n_ahead", args.n_ahead if not args.baseline else 1)
    n_ahead_talk = 1
    use_start_thought_token = params.get("use_start_thought_token", True)
    use_end_thought_token = params.get("use_end_thought_token", True)
    include_policy_loss = params.get("include_policy_loss", True)
    gumbel_detach = params.get("gumbel_detach", True)
    merged_talk_heads = params.get("merged_talk_heads", True)
    residual_think_head = params.get("residual_think_head", False)
    optimize_lm_head_only_at_start = params.get("optimize_lm_head_only_at_start", False)
    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        cache_dir=args.root_prefix + "cache",
        max_thoughts=n_ahead + n_ahead_talk + 1,
        merged_talk_heads=merged_talk_heads,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
    ).cuda()
    print(model.max_thoughts)
    print("Loaded model")
    tokenizer = LlamaTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    special_tokens_to_add = []
    special_tokens_to_add.append("<|startthought|>")
    special_tokens_to_add.append("<|endthought|>")
    # setattr(model, use_end_thought_token, True)
    # setattr(model, use_start_thought_token, True)
    # if model.use_start_thought_token:
    #     special_tokens_to_add.append("<|startthought|>")
    # if model.use_end_thought_token:
    #     special_tokens_to_add.append("<|endthought|>")
    if special_tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
    model.tokenizer = tokenizer
    model.gumbel_detach = gumbel_detach
    model.include_policy_loss = include_policy_loss
    model.use_end_thought_token = use_end_thought_token
    model.use_start_thought_token = use_start_thought_token
    model.n_ahead = n_ahead
    model.n_ahead_talk = n_ahead_talk
    model.n_passes = 1
    model.residual_think_head = residual_think_head
    if args.baseline:
        model.skip_residual = True
        model.cumulative_residual = False
        model.clever_residual = False
        model.base_residual = False
    model.optimize_lm_head_only_at_start = optimize_lm_head_only_at_start
    model.use_policy_loss = False
    model.rm_initialized = True
    model.first_run = False
    model.wandb_enabled = False
    model.config_params = params
    model.run_start = int(time.time())
    model.eval_mode = True
    model.eval()
    return model


def extract_first_integer(s):
    match = re.search(r'\d+', s.replace(',', ''))
    if match:
        return int(match.group())
    return None

# Initialize distributed environment
dist.init()
devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
torch.cuda.set_device(devices[0])

# Set random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

# Load the GSM8K dataset and the model
cot_dataset_gsm = load_dataset("gsm8k", "main", split="test", ignore_verifications=True).shuffle(seed=random_seed)
model = model_init(None)

start_question = args.device_batch_size * args.batch_idx
end_question = args.device_batch_size * (args.batch_idx + 5)

# Split data across processes if using distributed
if dist.size() > 1:
    # Calculate start and end indices for this process
    total_questions = min(args.max_idx, end_question) - start_question
    questions_per_process = total_questions // dist.size()
    start_question += questions_per_process * dist.rank()
    end_question = start_question + questions_per_process
    if dist.rank() == dist.size() - 1:  # Last process takes remaining questions
        end_question = min(args.max_idx, args.device_batch_size * (args.batch_idx + 5))

# Iterate over the questions for the current device
batch_size = 1
all_results = []
for batch_start in tqdm(range(start_question, min(args.max_idx, end_question), batch_size)):
    last_save_folder = f"answers/eval_{'baseline' if args.baseline else 'ft'}_{args.n_ahead if not args.baseline else 1}_{args.temp}_{args.n_votes}"
    extracted_answers = []
    for vote_idx in range(1, args.n_votes + 1):
        folder_name = f"answers/eval_{'baseline' if args.baseline else 'ft'}_{args.n_ahead if not args.baseline else 1}_{args.temp}_{vote_idx}"
        # Get the current batch of questions
        batch_questions = cot_dataset_gsm[batch_start:batch_start+batch_size]
        input_texts = ["Q: " + q + args.zero_shot_cot_prompt for q in batch_questions["question"]]
        input_ids = model.tokenizer(input_texts, return_tensors="pt", padding=True).to(model.device)
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        started_generating_answer_at = None
        
        # Generate the solution
        with torch.no_grad():
            finished_generating = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
            for cur_token_idx in range(args.start_final_answer_idx + args.answer_length):
                # Sample the next token
                new_ids = model(
                    input_ids[~finished_generating],
                    attention_mask=attention_mask[~finished_generating]
                )['logits']
                #changed
                if torch.isnan(new_ids).any():
                    print(new_ids)
                #print(new_ids.shape)
                # Mask out the start and end thought tokens so we don't accidentally sample them
                new_ids[:, :, model.tokenizer.vocab_size:] = -float("inf")
                for list_idx, answer_idx in enumerate((~finished_generating).nonzero(as_tuple=True)[0]):
                    # Find the index of the last token that is not padding
                    base_answer_ids = input_ids[answer_idx]
                    new_answer_ids = new_ids[list_idx]
                    last_token_idx = (base_answer_ids != model.tokenizer.pad_token_id).nonzero(as_tuple=True)[0].max()
                    if args.temp == 0:
                        new_ids_sampled = torch.argmax(new_answer_ids[last_token_idx]).unsqueeze(0)
                    else:
                        new_ids_sampled = torch.multinomial(torch.nn.functional.softmax(new_answer_ids[last_token_idx] / args.temp, dim=-1), 1)
                    # Assign the new id to the last token
                    if last_token_idx + 1 >= len(base_answer_ids):
                        # Add padding everywhere
                        new_padding = torch.full((len(input_ids), 1), model.tokenizer.pad_token_id, dtype=torch.long, device=input_ids.device)
                        input_ids = torch.cat([input_ids, new_padding], dim=-1)
                        attention_mask = torch.cat([attention_mask, torch.zeros_like(new_padding)], dim=-1)
                    attention_mask[answer_idx, last_token_idx + 1] = 1
                    input_ids[answer_idx, last_token_idx + 1] = new_ids_sampled
                    if new_ids_sampled == model.tokenizer.eos_token_id or new_ids_sampled == model.tokenizer.bos_token_id or new_ids_sampled == model.tokenizer.pad_token_id:
                        finished_generating[answer_idx] = 1
                    # "if "Q:" shows up multiple times, remove the last "Q:" and everything after it
                    decoded = model.tokenizer.decode(input_ids[answer_idx], skip_special_tokens=True)
                    end_strs = ["Q:", "\n\n\n"]
                    if any([decoded.count(end_str) > 1 for end_str in end_strs]):
                        # Get the first end_str that shows up in the decoded text multiple times
                        end_str = next(end_str for end_str in end_strs if decoded.count(end_str) > 1)
                        # Remove the last "Q:" and everything after it
                        decoded = decoded.split(end_str)[:-1]
                        new_answer = model.tokenizer.encode(decoded, return_tensors="pt").to(model.device)
                        input_ids[answer_idx] = torch.ones_like(input_ids[answer_idx]) * model.tokenizer.pad_token_id
                        input_ids[answer_idx, :new_answer.shape[1]] = new_answer
                        attention_mask[answer_idx] = (input_ids[answer_idx] != model.tokenizer.pad_token_id).long()
                        finished_generating[answer_idx] = 1

                # Check if we should start generating the final answer
                if (
                    (cur_token_idx == args.start_final_answer_idx and started_generating_answer_at is None) 
                    or finished_generating.all()
                ):
                    # If we haven't started generating the final answer yet, start now
                    if started_generating_answer_at is None:
                        finished_generating = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
                        started_generating_answer_at = cur_token_idx
                        # Append "Final Answer:" to the end of the generated text
                        base_texts = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                        final_texts = [text.rstrip() + args.final_answer_text for text in base_texts]
                        encoded_final_texts = model.tokenizer(final_texts, return_tensors="pt", padding=True).to(model.device)
                        attention_mask = encoded_final_texts.attention_mask
                        input_ids = encoded_final_texts.input_ids
                    else:
                        # We finished generating the answer
                        break
                
                if started_generating_answer_at is not None:
                    if cur_token_idx - started_generating_answer_at > args.answer_length:
                        break

        # Collect the generated answers for evaluation
        for i, encoded_final_text in enumerate(input_ids):
            question_idx = batch_start + i
            decoded_text = model.tokenizer.decode(encoded_final_text, skip_special_tokens=True)
            vote_extracted_number = decoded_text.split(args.final_answer_text)[-1]
            # Extract the first number from the answer text
            vote_extracted_number = extract_first_integer(vote_extracted_number)
            extracted_correct_answer = extract_first_integer(cot_dataset_gsm[question_idx]["answer"].split("#### ")[-1])
            extracted_answers.append((vote_extracted_number, extracted_correct_answer, decoded_text))

        # Process results for this batch
        extracted_number = Counter([extracted_number for extracted_number, _, _ in extracted_answers])
        extracted_most_common = extracted_number.most_common(1)[0][0]
        correct = extracted_most_common == extracted_answers[0][1]
        
        # Calculate pass@k
        unique_answers = set(extracted_number for extracted_number, _, _ in extracted_answers)
        pass_at_k = any(ans == extracted_answers[0][1] for ans in unique_answers)
        
        # Print progress
        print(f"Majority Vote Question {batch_start} - Correct: {correct} - Extracted: {extracted_number} - True: {extracted_correct_answer}")
        print(f"Pass@k Question {batch_start} - Correct: {pass_at_k}")
        
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # Store results
    result = {
        'question_idx': batch_start,
        'all_answers': [extracted_number for extracted_number, _, _ in extracted_answers],
        'majority_vote_answer': extracted_most_common,
        'pass@k_answer': pass_at_k,
        'true_answer': extracted_answers[0][1],
        'decoded_texts': [decoded for _, _, decoded in extracted_answers],
        'vote_folder': folder_name
    }
    all_results.append(result)


def save_result(result):
    """Helper function to save a single result"""
    # Save to vote-specific folder
    save_filename = f"{result['vote_folder']}/{result['question_idx']}.txt"
    joined_final_texts = ("\n" + "=" * 100 + "\n").join(result['decoded_texts'])
    with open(save_filename, "w") as f:
        f.write(joined_final_texts + "\n" + 
               "Majority Vote: " + str(result['majority_vote_answer']) + "\n" + 
               "Pass@k: " + str(result['pass@k_answer']) + "\n" + 
               "True: " + str(result['true_answer']))

# Gather and save results
if dist.size() > 1:
    all_results = dist.gather(all_results, dst=0)
    if dist.is_main():
        # Flatten gathered results
        all_results = list(itertools.chain(*all_results))
        # Sort results by question index
        all_results.sort(key=lambda x: x['question_idx'])
        # Save all results
        for result in all_results:
            save_result(result)
        
        # Calculate accuracies
        total = len(all_results)
        majority_correct = sum(1 for r in all_results if r['majority_vote_answer'] == r['true_answer'])
        pass_k_correct = sum(1 for r in all_results if r['pass@k_answer'])
        
        majority_acc = majority_correct / total
        pass_k_acc = pass_k_correct / total
        
        # Save consolidated results and accuracies
        final_results = {
            'raw_results': all_results,
            'majority_vote_accuracy': majority_acc,
            'pass_at_k_accuracy': pass_k_acc,
            'num_questions': total
        }
        print(f"Majority Vote Accuracy: {majority_acc} Pass@k Accuracy: {pass_k_acc}")
        
        json_filename = f"{result['vote_folder']}/results.json"
        with open(json_filename, 'w') as f:
            json.dump(final_results, f)
else:
    # Save results directly if not using distributed
    for result in all_results:
        save_result(result)
        
    # Calculate accuracies
    total = len(all_results)
    majority_correct = sum(1 for r in all_results if r['majority_vote_answer'] == r['true_answer']) 
    pass_k_correct = sum(1 for r in all_results if r['pass@k_answer'])
    
    majority_acc = majority_correct / total
    pass_k_acc = pass_k_correct / total
    
    # Save consolidated results and accuracies
    final_results = {
        'raw_results': all_results,
        'majority_vote_accuracy': majority_acc, 
        'pass_at_k_accuracy': pass_k_acc,
        'num_questions': total
    }
    print(f"Majority Vote Accuracy: {majority_acc} Pass@k Accuracy: {pass_k_acc}")
    
    json_filename = f"{result['vote_folder']}/results.json"
    with open(json_filename, 'w') as f:
        json.dump(final_results, f)
