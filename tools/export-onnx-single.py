from typing import List, Optional
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import sys

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
PROMPT = PROMPT_DICT['prompt_no_input']

CACHE_DIR = 'alpaca_out'

class LLAMA():
    def __init__(self, outdir):
        self.device = 'cpu'
        self.model = LlamaForCausalLM.from_pretrained(outdir, cache_dir=CACHE_DIR, local_files_only=True)
        self.model.to(self.device)
        print('model loaded!')
        self.tokenizer = LlamaTokenizer.from_pretrained(outdir, cache_dir=CACHE_DIR, local_files_only=True)
        print('tokenizer loaded!')
    def export(
            self,
            prompt: str = "bonjour",
            n: int = 1,
            total_tokens: int = 2000,
            temperature: float = 0.1, 
            top_p: float = 1.0,
            repetition_penalty: float = 1) -> List[str]:

        format_prompt = PROMPT.format_map({'instruction': prompt})
        _input = self.tokenizer(format_prompt, return_tensors="pt").input_ids.to(self.device)
        print('input ids created,now converting to onnx--------->')
        with torch.no_grad():
            symbolic_names = {0: "batch_size", 1: "seq_len"}
            torch.onnx.export(
                self.model,
                _input,
                "llama-7b.onnx",
                input_names=["input_ids"],
                output_names=["output"],
                dynamic_axes={"input_ids": symbolic_names, "output": symbolic_names},
                opset_version=15,  # Use a suitable opset version, such as 12 or 13
            )
        # outputs = self.model.generate(
        #     _input,
        #     num_return_sequences=n,
        #     max_length=total_tokens,
        #     do_sample=True,
        #     temperature=temperature,
        #     top_p=top_p,
        #     top_k=40,
        #     repetition_penalty=repetition_penalty
        # )
        # print('generation done, decoding output---------->')
        # out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # removing prompt b/c it's returned with every input 
        # out = [val.split('Response:')[1] for val in out]
        # print('Q: {} A: {}'.format(prompt, out))
        return None

x = LLAMA(sys.argv[1])
x.export()