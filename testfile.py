from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch

class T5KeyTermsGenerationPipeline:
    def __init__(self, model_name_or_path='t5-base'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, legacy = False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

    def generate_key_terms(self, text, max_length=50, min_length=10):
        input_text = "question: what are the complex terms in the passage? context: " + text
        #input_text = "extract key terms: " + text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=5000, truncation=True)
        key_terms_ids = self.model.generate(input_ids, max_length=max_length, min_length=min_length, num_beams=2, early_stopping=True)
        key_terms = self.tokenizer.decode(key_terms_ids[0], skip_special_tokens=True)
        key_terms_list = [term.strip() for term in key_terms.split(",")]
        return key_terms_list

    def summarize_key_terms(self, key_terms_list, max_length=100):
        summaries = []
        for term in key_terms_list:
            summary_prompt = f"Define: '{term}' "
            input_ids = self.tokenizer.encode(summary_prompt, return_tensors="pt", max_length=512, truncation=True)
            output_ids = self.model.generate(input_ids, max_length=max_length, num_beams=2, early_stopping=True)
            summary_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            summaries.append(summary_text.strip())
        return summaries  
    
    # Load the JSON file into a pandas DataFrame
df = pd.read_csv("projectstuff/simpletext-task2-train-input.tsv", delimiter='\t', encoding='utf-8')

# Display the DataFrame
print("DataFrame:")
print(df)

# Example usage:
text = "Boats, vessels designed for navigating water bodies, come in various forms such as rowboats, sailboats, and motorboats. Propulsion methods, including oars, sails, or engines, are utilized to maneuver boats through water, with hulls providing buoyancy and stability."
key_terms_generator = T5KeyTermsGenerationPipeline()
key_terms = key_terms_generator.generate_key_terms(text)
print("Key Terms:", key_terms)


key_terms = [
    "acquiesce",
    "benevolent",
    "cogitate",
    "deleterious",
    "ebullient",
    "facetious",
    "garrulous",
    "hedonistic",
    "impecunious",
    "juxtapose"
]
# Summarize the key terms
summaries = key_terms_generator.summarize_key_terms(key_terms)
for term, summary in zip(key_terms, summaries):
    print(f"{term}: {summary}")

