from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from utils import Utils
import yaml

class LLMForTestGeneration:
    def __init__(self, model_path, max_model_len, sample_size, max_tokens, temperature,
                 gpu_memory_utilization=0.9, enable_prefix_caching=True, max_num_seqs=128):
        
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching

        self.sample_size = sample_size
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.llm = LLM(
            model=model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            max_num_seqs=128
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.llm.set_tokenizer(self.tokenizer)

        print("------------------------LLMForTestGeneration Arguments------------------------")
        print(f"model_path: {model_path}")
        print(f"max_model_len: {max_model_len}")
        print(f"gpu_memory_utilization: {gpu_memory_utilization}")
        print(f"enable_prefix_caching: {enable_prefix_caching}")
        print(f"sample_size: {sample_size}")
        print(f"max_tokens: {max_tokens}")
        print(f"temperature: {temperature}")
        print(f"max_num_seqs: {max_num_seqs}")
        print("-----------------------------------------------------------------------------")
    
    def base_model_generate(self, data, batch_size=50):
        print("-------------------------Input Prompt Examples-------------------------")
        print(data[0]['prompt'])
        HumanEval_for_generation = []
        
        for idx in tqdm(range(0, len(data))):
            prompt = data[idx]['prompt']
            all_samples = []
            # 将sample_size分成多个batch处理
            for start_idx in range(0, self.sample_size, batch_size):
                # 计算当前batch的实际大小
                current_batch_size = min(batch_size, self.sample_size - start_idx)
                expanded_prompt = [prompt] * current_batch_size
                outputs = self.llm.generate(expanded_prompt, self.sampling_params, use_tqdm=False)
                batch_samples = [output.outputs[0].text for output in outputs]
                all_samples.extend(batch_samples)
                
            HumanEval_for_generation.append({
                "prompt": prompt,
                "samples": all_samples
            })
        return HumanEval_for_generation
    
    def chat_model_generate(self, data):
        print("-------------------------Input Prompt Examples-------------------------")
        print(data[0]['prompt'])
        HumanEval_for_generation = []
        for idx in tqdm(range(0, len(data))):
            prompt = data[idx]['prompt']
            chat = [
                {"role": "system", "content": "You are an AI assistant specialized in generating code."},
                {"role": "user", "content": prompt}
            ]
            chat = self.tokenizer.apply_chat_template(chat, tokenize=False)
            expanded_prompt = [chat] * self.sample_size
            outputs = self.llm.generate(expanded_prompt, self.sampling_params, use_tqdm=False)
            samples = [output.outputs[0].text for output in outputs]
            HumanEval_for_generation.append({
                **data[idx],
                "samples": samples
            })
        return HumanEval_for_generation
    
def test_generation(data_path, model_path, output_path, sample_size, max_tokens, temperature=0.8,
                    gpu_memory_utilization=0.9, enable_prefix_caching=True, max_model_len=4096, batch_size=50, max_num_seqs=128):
    # read the data
    data = Utils.load_jsonl_file(data_path)
    
    llm_for_test_generation = LLMForTestGeneration(
        model_path=model_path, max_model_len=max_model_len, 
        sample_size=sample_size, max_tokens=max_tokens, temperature=temperature,
        gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=enable_prefix_caching, max_num_seqs=max_num_seqs
    )
    HumanEval_for_generation = llm_for_test_generation.base_model_generate(data, batch_size=batch_size)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    Utils.save_jsonl_file(HumanEval_for_generation, output_path)
  
def main(data_path, model_path, output_path, sample_size, max_tokens, temperature=0.8,
         gpu_memory_utilization=0.9, enable_prefix_caching=True, max_model_len=4096, batch_size=50, max_num_seqs=128):
    test_generation(data_path, model_path, output_path, sample_size, max_tokens, temperature,
                    gpu_memory_utilization, enable_prefix_caching, max_model_len, batch_size, max_num_seqs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the data file", default="./dataset/HumanEval_for_code_generation.jsonl")
    parser.add_argument("--model_path", type=str, help="Path to the model", default="/home/clw/hf_local_models/Llama-3-8B-Instruct")
    parser.add_argument("--output_path", type=str, help="Path to save the generated data", default="./generated_data/generated_test_cases/HumanEval")

    parser.add_argument("--sample_size", type=int, help="Number of samples to generate", default=100)
    parser.add_argument("--max_tokens", type=int, help="Maximum tokens to generate", default=300)
    parser.add_argument("--temperature", type=float, help="Temperature for sampling", default=0.8)
    parser.add_argument("--gpu_memory_utilization", type=float, help="GPU memory utilization", default=0.9)
    parser.add_argument("--enable_prefix_caching", type=bool, help="Enable prefix caching", default=True)
    parser.add_argument("--max_model_len", type=int, help="Maximum model length", default=4096)
    parser.add_argument("--batch_size", type=int, help="Batch size for generation", default=50)
    parser.add_argument("--max_num_seqs", type=int, help="Maximum number of sequences", default=128)

    args = parser.parse_args()
    
    # log and format the arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    main(args.data_path, args.model_path, args.output_path, args.sample_size, args.max_tokens, 
         args.temperature, args.gpu_memory_utilization, args.enable_prefix_caching, args.max_model_len,
         args.batch_size, args.max_num_seqs)