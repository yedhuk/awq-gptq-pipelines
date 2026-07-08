
from huggingface_hub import snapshot_download
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

class GPTQPipeline:
    def __init__(self, model_path, tokenizer, bits, group_size):
        # self.model = model
        # 
        self.config = QuantizeConfig(bits=bits, group_size=group_size)
        self.model = GPTQModel.load(model_path, self.config)
        self.tokenizer = tokenizer      
        
    def generate(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def quantize(self, calibration_dataset, batch=2):
        self.model.quantize( calibration_dataset, batch_size=batch)
    
    def save_quantized(self, save_path):
        self.model.save_quantized(save_path)
        self.tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    model_id = 'meta-llama/Llama-3.2-3B-Instruct'
    quant_path = './models/llama-3.2-3b-instruct-gptq-custom'

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # manual download of the model to avoid downloading the original weights
    local_model_path = snapshot_download(
    repo_id=model_id,
    ignore_patterns=["original/*", "*.pth"]
)
    bits = 4
    group_size = 128
    pipeline = GPTQPipeline(local_model_path, tokenizer, bits, group_size)
    result = pipeline.generate("What is the capital of France?")
    print(result)