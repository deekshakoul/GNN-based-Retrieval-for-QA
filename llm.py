from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class LLMGenerator:
    """
    Answer generator using a pre-trained T5 model
    """
    def __init__(self, model_name="t5-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
    
    def generate(self, question, contexts, max_length=100):
        """
        Generate an answer given a question and retrieved contexts
        """
        context_text = " ".join(contexts)
    
        input_text = f"question: {question} context: {context_text}"
        
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer