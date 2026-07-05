from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer



class AWQPipeline:
    def __init__(self, base_model_path: str, config: dict = None):
        self.model = AutoAWQForCausalLM.from_pretrained(base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.default_quant_config = { 
            "zero_point": True, 
            "q_group_size": 128, 
            "w_bit": 4, 
            "version": "GEMM" 
        }
        self.config = config if config is not None else self.default_quant_config

    def quantize(self, quant_config: dict):
        self.model.quantize(self.tokenizer, quant_config=quant_config)



    def generate(self, prompt: str, max_length: int = 50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def save_quantized(self, save_path: str):
        self.model.save_quantized(save_path)
        self.tokenizer.save_pretrained(save_path)

if __name__ == "__main__":

    model_path = 'meta-llama/Llama-3.2-3B-Instruct'
    quant_path = './models/llama-3.2-3b-instruct-awq-custom'
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,            
        "w_bit": 4,
        "version": "GEMM"
    }   
    pipeline = AWQPipeline(model_path, quant_config, quant_path)
    result = pipeline.generate("What is the capital of France?")
    print(result)