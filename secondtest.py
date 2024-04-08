from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

class T5KeyTermsGenerationPipeline:
    def __init__(self, model_name_or_path="google-t5/t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

    def generate_key_terms(self, text, max_length=50, min_length=10):
        input_text = "question: What are 5 keywords in this passage? context: " + text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=5000, truncation=True)
        key_terms_ids = self.model.generate(input_ids, max_length=max_length, min_length=min_length, num_beams=2, early_stopping=True)
        key_terms = self.tokenizer.decode(key_terms_ids[0], skip_special_tokens=True)
        key_terms_list = [term.strip() for term in key_terms.split(",")]
        return key_terms_list

    def define_key_terms(self, key_terms_list, max_length=100):
        definitions = []
        for term in key_terms_list:
            summary_prompt = f"summerize: '{term}'"
            input_ids = self.tokenizer.encode(summary_prompt, return_tensors="pt", max_length=512, truncation=True)
            output_ids = self.model.generate(input_ids, max_length=max_length, num_beams=2, early_stopping=True)
            definition_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            definitions.append(definition_text.strip())
        return definitions  

# Load the TSV file into a pandas DataFrame
df = pd.read_csv("projectstuff/simpletext-task2-train-input.tsv", delimiter='\t', encoding='utf-8')

# Extract keywords from the 5th column of lines 2 to 6
for i in range(1, 6):  # Loop through lines 2 to 6
    keywords = df.iloc[i, 4].split()
    print(f"Output {i}:")
    print("Complex Terms:", keywords)
    # Example usage:
    key_terms_generator = T5KeyTermsGenerationPipeline()
    # Generate definitions for the extracted keywords
    definitions = key_terms_generator.define_key_terms(keywords)
    for keyword, definition in zip(keywords, definitions):
        print(f"{keyword}: {definition}")
    print()