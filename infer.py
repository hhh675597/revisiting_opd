#!/usr/bin/env python3
"""
Inference script that strictly follows the math environment pipeline:
1. Load a math problem from dataset
2. Apply environment-specific prompt from math.py
3. Load math-specific teacher model (OpenThinker3-7B)
4. Perform inference and show complete tokenization
5. Output results as JSON
"""

import os
import json
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Any
from functools import partial

# Import the math template from the environment prompts
from agent_system.environments.prompts.math import MATH_TEMPLATE
from agent_system.environments.env_package.math.envs import build_math_envs
from agent_system.environments.env_manager import MathEnvironmentManager


def math_projection(text_actions: List[str]) -> tuple:
    """Simple projection function for math actions"""
    actions = text_actions
    valids = [True] * len(text_actions)
    return actions, valids


class MathInferencePipeline:
    """Complete inference pipeline following the exact environment flow"""
    
    def __init__(self, model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path = data_path
        
        print(f"[1/4] Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        print(f"[2/4] Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        
        print(f"[3/4] Loading dataset from {data_path}...")
        self.dataset = datasets.load_dataset(
            "parquet",
            data_files=data_path,
            split="train"
        )
        
        print(f"[4/4] Initializing environment...")
        # Create a single environment for inference
        self._envs = build_math_envs(seed=0, env_num=1, group_n=1, is_train=False)
        projection_f = partial(math_projection)
        
        # Create a minimal config object
        class MinimalConfig:
            class env:
                seed = 0
        
        self.env_manager = MathEnvironmentManager(self._envs, projection_f, MinimalConfig())
        
        print("\n✓ Pipeline initialized successfully!\n")
    
    def load_math_problem(self, idx: int = 0) -> Dict[str, Any]:
        """Load a math problem from the dataset"""
        example = self.dataset[idx]
        
        # Extract the raw question from the prompt structure
        raw_question = example['prompt'][0]['content']
        ground_truth = example['reward_model']['ground_truth']
        data_source = example.get('data_source', 'unknown')
        
        return {
            'raw_question': raw_question,
            'ground_truth': ground_truth,
            'data_source': data_source,
            'full_example': example
        }
    
    def apply_env_prompt(self, raw_question: str) -> str:
        """Apply the environment-specific prompt template from math.py"""
        # This follows the exact flow in MathEnvironmentManager.build_text_obs
        formatted_prompt = MATH_TEMPLATE.format(task_description=raw_question)
        return formatted_prompt
    
    def apply_chat_template(self, env_prompt: str) -> str:
        """Apply the model's chat template"""
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": env_prompt}
        ]
        
        # Apply chat template
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return formatted_text
    
    def tokenize_and_analyze(self, text: str) -> Dict[str, Any]:
        """Tokenize the text and return detailed token information"""
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Get token strings
        token_strings = [self.tokenizer.decode([tok]) for tok in tokens]
        
        # Create detailed token info
        token_details = []
        for idx, (token_id, token_str) in enumerate(zip(tokens, token_strings)):
            token_details.append({
                'position': idx,
                'token_id': int(token_id),
                'token_string': token_str,
                'token_repr': repr(token_str)  # Shows escape sequences
            })
        
        return {
            'token_ids': [int(t) for t in tokens],
            'token_strings': token_strings,
            'token_details': token_details,
            'total_tokens': len(tokens)
        }
    
    def generate_response(self, text: str, max_new_tokens: int = 2048) -> Dict[str, Any]:
        """Generate response from the model"""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        
        print(f"Generating response (max_new_tokens={max_new_tokens})...")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,  # Greedy decoding
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract only the generated tokens (exclude input)
        generated_ids = outputs[0][input_length:].cpu().tolist()
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # Get token details for generated tokens
        generated_token_strings = [self.tokenizer.decode([tok]) for tok in generated_ids]
        
        generated_token_details = []
        for idx, (token_id, token_str) in enumerate(zip(generated_ids, generated_token_strings)):
            generated_token_details.append({
                'position': idx,
                'token_id': int(token_id),
                'token_string': token_str,
                'token_repr': repr(token_str)
            })
        
        return {
            'generated_text': generated_text,
            'generated_token_ids': generated_ids,
            'generated_token_strings': generated_token_strings,
            'generated_token_details': generated_token_details,
            'total_generated_tokens': len(generated_ids)
        }
    
    def run_inference(self, problem_idx: int = 0, max_new_tokens: int = 2048) -> Dict[str, Any]:
        """Run complete inference pipeline"""
        
        print("="*80)
        print("MATH INFERENCE PIPELINE")
        print("="*80)
        
        # Step 1: Load math problem
        print(f"\n[STEP 1] Loading math problem (index={problem_idx})...")
        problem_data = self.load_math_problem(problem_idx)
        print(f"✓ Loaded problem from data_source: {problem_data['data_source']}")
        print(f"✓ Ground truth: {problem_data['ground_truth']}")
        
        # Step 2: Apply environment prompt
        print(f"\n[STEP 2] Applying environment-specific prompt (from math.py)...")
        env_prompt = self.apply_env_prompt(problem_data['raw_question'])
        print(f"✓ Applied MATH_TEMPLATE")
        
        # Step 3: Apply chat template
        print(f"\n[STEP 3] Applying model chat template...")
        full_prompt = self.apply_chat_template(env_prompt)
        print(f"✓ Applied chat template")
        
        # Step 4: Tokenize and analyze prompt
        print(f"\n[STEP 4] Tokenizing prompt...")
        prompt_tokens = self.tokenize_and_analyze(full_prompt)
        print(f"✓ Tokenized: {prompt_tokens['total_tokens']} tokens")
        
        # Step 5: Generate response
        print(f"\n[STEP 5] Generating response with teacher model...")
        response_data = self.generate_response(full_prompt, max_new_tokens)
        print(f"✓ Generated: {response_data['total_generated_tokens']} tokens")
        
        # Compile complete results
        results = {
            'pipeline_steps': {
                'step1_raw_problem': {
                    'raw_question': problem_data['raw_question'],
                    'ground_truth': problem_data['ground_truth'],
                    'data_source': problem_data['data_source']
                },
                'step2_env_prompt': {
                    'template_used': 'MATH_TEMPLATE from agent_system.environments.prompts.math',
                    'formatted_prompt': env_prompt
                },
                'step3_chat_template': {
                    'full_prompt': full_prompt
                },
                'step4_prompt_tokenization': prompt_tokens,
                'step5_generation': response_data
            },
            'model_info': {
                'model_path': self.model_path,
                'model_name': os.path.basename(self.model_path),
                'tokenizer_vocab_size': self.tokenizer.vocab_size
            },
            'summary': {
                'problem_index': problem_idx,
                'data_source': problem_data['data_source'],
                'ground_truth': problem_data['ground_truth'],
                'prompt_tokens': prompt_tokens['total_tokens'],
                'generated_tokens': response_data['total_generated_tokens'],
                'total_tokens': prompt_tokens['total_tokens'] + response_data['total_generated_tokens']
            }
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a human-readable summary"""
        print("\n" + "="*80)
        print("INFERENCE SUMMARY")
        print("="*80)
        
        summary = results['summary']
        print(f"\nProblem Index: {summary['problem_index']}")
        print(f"Data Source: {summary['data_source']}")
        print(f"Ground Truth: {summary['ground_truth']}")
        print(f"\nTokenization:")
        print(f"  - Prompt tokens: {summary['prompt_tokens']}")
        print(f"  - Generated tokens: {summary['generated_tokens']}")
        print(f"  - Total tokens: {summary['total_tokens']}")
        
        print(f"\n" + "-"*80)
        print("RAW QUESTION:")
        print("-"*80)
        print(results['pipeline_steps']['step1_raw_problem']['raw_question'][:500])
        
        print(f"\n" + "-"*80)
        print("ENV PROMPT (with MATH_TEMPLATE):")
        print("-"*80)
        print(results['pipeline_steps']['step2_env_prompt']['formatted_prompt'][:500])
        
        print(f"\n" + "-"*80)
        print("GENERATED RESPONSE:")
        print("-"*80)
        print(results['pipeline_steps']['step5_generation']['generated_text'][:1000])
        
        print(f"\n" + "-"*80)
        print("FIRST 10 PROMPT TOKENS:")
        print("-"*80)
        for token in results['pipeline_steps']['step4_prompt_tokenization']['token_details'][:10]:
            print(f"  [{token['position']:4d}] ID={token['token_id']:6d} | {token['token_repr']}")
        
        print(f"\n" + "-"*80)
        print("FIRST 20 GENERATED TOKENS:")
        print("-"*80)
        for token in results['pipeline_steps']['step5_generation']['generated_token_details'][:20]:
            print(f"  [{token['position']:4d}] ID={token['token_id']:6d} | {token['token_repr']}")
        
        print("\n" + "="*80)


def main():
    # Configuration
    MATH_TEACHER = "/data/home/zdhs0010/agentic/model/OpenThinker3-7B"
    DATA_PATH = "/data/home/zdhs0010/agentic/verl-agent-multi/data/math_opd/test.parquet"
    OUTPUT_PATH = "math_inference_output.json"
    
    # Problem index to test (change this to test different problems)
    PROBLEM_IDX = 0
    MAX_NEW_TOKENS = 16384
    
    # Initialize pipeline
    pipeline = MathInferencePipeline(
        model_path=MATH_TEACHER,
        data_path=DATA_PATH
    )
    
    # Run inference
    results = pipeline.run_inference(
        problem_idx=PROBLEM_IDX,
        max_new_tokens=MAX_NEW_TOKENS
    )
    
    # Save results
    pipeline.save_results(results, OUTPUT_PATH)
    
    # Print summary
    pipeline.print_summary(results)
    
    print(f"\n✓ Complete! Full results saved to: {OUTPUT_PATH}")
    print(f"  - Contains detailed token-by-token information")
    print(f"  - Shows complete pipeline from raw problem to generated response")


if __name__ == "__main__":
    main()
