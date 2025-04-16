import argparse
import random
import uuid
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning Script")
    parser.add_argument('--model', type=str, required=True,
                        help="Pretrained model name or path (e.g., 'gpt2' or a local path)")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name or path (e.g., 'databricks/databricks-dolly-15k')")
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help="Learning rate for the optimizer")
    parser.add_argument('--epochs', type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for training")
    parser.add_argument('--gradient_accumulation', type=int, default=8,
                        help="how many steps you accumulate to form a 'large batch'.")
    parser.add_argument('--save_path', type=str, help="path to save the model checkpoint")
    parser.add_argument('--max_length', type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument('--beta', type=float, default=0.01,
                        help="Beta parameter for DPO loss")

    args = parser.parse_args()

    # Load tokenizer and model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Ensure a padding token is defined (for models that don't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.train()

    # Load dataset (using the "train" split)
    dataset = load_dataset(args.dataset)["train"]

    # Define a tokenization function that masks out the loss on the prompt
    def tokenize_fn(example):
        # Tokenize the prompt (instructions) and response separately.
        
        instruction = example["prompt"]
        chosen_response = example["chosen"][-1]["content"]
        rejected_response = example["rejected"][-1]["content"]
        id = str(uuid.uuid4())

        # Use add_special_tokens=False so we can control token concatenation
        instr_tokens = tokenizer(instruction, truncation=True, max_length=args.max_length, add_special_tokens=False)
        chosen_resp_tokens = tokenizer(chosen_response, truncation=True, max_length=args.max_length, add_special_tokens=False)
        rejected_resp_tokens = tokenizer(rejected_response, truncation=True, max_length=args.max_length, add_special_tokens=False)

        # Tokenize a separator (here we use "\n\n")
        sep_tokens = tokenizer("\n\n", add_special_tokens=False)["input_ids"]

        # TODO: Concatenate: [instruction] + [separator] + [response], you would need to do it seperately for chosen and rejected input ids
        chosen_input_ids = instr_tokens["input_ids"] + sep_tokens + chosen_resp_tokens["input_ids"]
        rejected_input_ids = instr_tokens["input_ids"] + sep_tokens + rejected_resp_tokens["input_ids"]
        # Create labels: mask out (with -100) the tokens corresponding to the instruction and separator, again you need to do this for both chosen and rejected
        mask_labels = [-100] * (len(instr_tokens["input_ids"]) + len(sep_tokens))
        chosen_labels = mask_labels + chosen_resp_tokens["input_ids"]
        rejected_labels = mask_labels + rejected_resp_tokens["input_ids"]

        # Then trunctate the inputs / pad the inputs according to args.max_length
        if len(chosen_input_ids) > args.max_length:
            # Truncate
            chosen_input_ids = chosen_input_ids[:args.max_length]
            chosen_labels = chosen_labels[:args.max_length]
        else:
            # Pad
            padding_length = args.max_length - len(chosen_input_ids)
            chosen_input_ids = chosen_input_ids + [tokenizer.pad_token_id] * padding_length
            chosen_labels = chosen_labels + [-100] * padding_length
        
        if len(rejected_input_ids) > args.max_length:
            # Truncate
            rejected_input_ids = rejected_input_ids[:args.max_length]
            rejected_labels = rejected_labels[:args.max_length]
        else:
            # Pad
            padding_length = args.max_length - len(rejected_input_ids)
            rejected_input_ids = rejected_input_ids + [tokenizer.pad_token_id] * padding_length
            rejected_labels = rejected_labels + [-100] * padding_length
            
        # Create attention mask
        chosen_attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in chosen_input_ids]
        rejected_attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in rejected_input_ids]

        # Your code ends here.

        return {"id": id, "chosen_input_ids": chosen_input_ids, "chosen_attention_mask": chosen_attention_mask, "chosen_labels": chosen_labels,
                "rejected_input_ids": rejected_input_ids, "rejected_attention_mask": rejected_attention_mask, "rejected_labels": rejected_labels}

    tokenized_dataset = dataset.map(tokenize_fn, batched=False)
    tokenized_dataset.set_format(type='torch', 
        columns=['id', 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels'])
    
    samples = [random.randint(0, len(dataset) - 1) for _ in range(3)]
    
    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # TODO: Function to compute log probabilities for tokens where labels are not -100
    def compute_token_logprobs(logits, input_ids, labels):
        """Compute log probabilities for tokens where labels are not -100"""
        
        # Create log probabilities from logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Your code ends here.

        batch_size = logits.size(0)
        seq_length = logits.size(1)
        vocab_size = logits.size(2)
        
        # Initialize result tensor
        token_logprobs = torch.zeros(batch_size, seq_length, device=logits.device)
        
        # For each element in the batch, get the positions where labels are not -100
        for i in range(batch_size):
            # Find positions where labels are not -100 (i.e., response tokens)
            response_mask = (labels[i] != -100)
            
            # Get the probability of the token
            for j in range(seq_length):
                if response_mask[j]:
                    token_id = input_ids[i, j]
                    token_logprobs[i, j] = log_probs[i, j, token_id]
            

        # Mask to only include response tokens
        mask = (labels != -100).float()
        masked_logprobs = token_logprobs * mask
        # Sum log probs and divide by number of tokens to get average
        batch_logprobs = masked_logprobs.sum(dim=1) / mask.sum(dim=1)
        
        # Your code ends here.

        return batch_logprobs
    
    id2logprobs = {}
    print("Calculating reference log probabilities...")
    
    # TODO: First, loop over the data using the model to get the chosen and rejected log probability
    model.eval()  # Set to eval mode for reference computation
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            print(f"Processing batch {index+1}/{len(dataloader)}", end="\r")
            
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            
            # Get the logits for the chosen responses
            chosen_outputs = model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                labels=chosen_labels
            )
            chosen_logits = chosen_outputs['logits']
            # Get the logits for the rejected responses
            rejected_outputs = model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                labels=rejected_labels
            )
            rejected_logits = rejected_outputs['logits']
            # Compute the log probabilities
            chosen_log_probs = compute_token_logprobs(chosen_logits, chosen_input_ids, chosen_labels)
            rejected_log_probs = compute_token_logprobs(rejected_logits, rejected_input_ids, rejected_labels)
        
            # Store the log probabilities in a dictionary
            for i in range(len(batch["id"])):
                id = batch["id"][i]
                id2logprobs[id] = {
                    "chosen_log_probs": chosen_log_probs[i].detach(),
                    "rejected_log_probs": rejected_log_probs[i].detach(),
                }
    
    print("\nReference log probabilities computed.")
    model.train()  # Set back to train mode
    
    # TODO: DPO Training loop
    beta = args.beta  # Beta parameter for DPO loss
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        print("\nSample generations:")
        for sample_idx in samples:
            sample = dataset[sample_idx]
            prompt = sample["prompt"]
            print(f"\nPrompt: {prompt}")
            # Tokenize the prompt (without response)
            input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
            # TODO: paste the code in section 3.1
            outputs = model.generate(
                input_ids=input_tokens["input_ids"],
                attention_mask=input_tokens["attention_mask"],
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")
        
        total_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        
        for index, batch in enumerate(dataloader):
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            
            # Get the reference log probabilities
            ref_log_probs = [id2logprobs[id] for id in batch["id"]]
            ref_chosen_log_probs = torch.stack([item["chosen_log_probs"] for item in ref_log_probs])
            ref_rejected_log_probs = torch.stack([item["rejected_log_probs"] for item in ref_log_probs])
            # Compute the logits for the chosen responses
            chosen_outputs = model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask
            )
            chosen_logits = chosen_outputs['logits']
            # Compute the logits for the rejected responses
            rejected_outputs = model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask
            )
            rejected_logits = rejected_outputs['logits']
            # Compute token log probabilities
            # Hint: you can call the helper function compute_token_logprobs
            policy_chosen_log_probs = compute_token_logprobs(chosen_logits, chosen_input_ids, chosen_labels)
            policy_rejected_log_probs = compute_token_logprobs(rejected_logits, rejected_input_ids, rejected_labels)
            
            # Compute DPO loss cacluation
            # The DPO loss: -log(σ(β(log_prob_difference(x_w) - log_prob_difference(x_l))))
            # Make sure to divide the loss by number of gradient accumulation steps
            chosen_log_ratio = policy_chosen_log_probs - ref_chosen_log_probs
            rejected_log_ratio = policy_rejected_log_probs - ref_rejected_log_probs
            
            logits = beta * (chosen_log_ratio - rejected_log_ratio)
            loss = -torch.nn.functional.logsigmoid(logits).mean()
            loss = loss / args.gradient_accumulation
            loss.backward()
            total_loss += loss.item()
                
            # Gradient Accumulation
            if (index + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f"Batch {index+1}/{len(dataloader)}, Loss: {total_loss:.4f}")
                total_loss = 0.0
            # Your code ends here.

        # Handle any remaining gradients at the end of epoch
        if total_loss > 0:
            optimizer.step()
            optimizer.zero_grad()
            
        
        # Optional: Save checkpoint at the end of each epoch
        checkpoint_path = f"{args.save_path}_epoch{epoch+1}"  
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    
    # Save the final fine-tuned model and tokenizer
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

if __name__ == "__main__":
    main()
