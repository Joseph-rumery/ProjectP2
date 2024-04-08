from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

class T5KeyTermsGenerationPipeline:
    def __init__(self, model_name_or_path='t5-base'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

    def generate_key_terms(self, keywords, max_length=50, min_length=10):
        input_text = "question: identify complex terms: " + ", ".join(keywords)
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=5000, truncation=True)
        key_terms_ids = self.model.generate(input_ids, max_length=max_length, min_length=min_length, num_beams=2, early_stopping=True)
        key_terms = self.tokenizer.decode(key_terms_ids[0], skip_special_tokens=True)
        key_terms_list = [term.strip() for term in key_terms.split(",")]
        return key_terms_list

    def define_key_terms(self, key_terms_list, max_length=100):
        definitions = []
        for term in key_terms_list:
            summary_prompt = f"Define '{term}'"
            input_ids = self.tokenizer.encode(summary_prompt, return_tensors="pt", max_length=512, truncation=True)
            output_ids = self.model.generate(input_ids, max_length=max_length, num_beams=2, early_stopping=True)
            definition_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            definitions.append(definition_text.strip())
        return definitions  
    
    def check_and_define_key_terms(self, line_data, max_length=100):
        word = line_data.iloc[1]  # Word in the 2nd column of the line
        key_terms = line_data.iloc[4].split()  # Keywords from the 5th column of the line
        
        if any(word in term for term in key_terms):
            definitions = self.define_key_terms([word], max_length=max_length)
            return word, definitions[0]
        else:
            return None, None

def process_tsv_file(file_path):
    # Load the TSV file into a pandas DataFrame
    df = pd.read_csv(file_path, delimiter='\t', encoding='utf-8')

    key_terms_generator = T5KeyTermsGenerationPipeline()
    
    # Extract and process lines 2 to 6
    for i in range(1, 100):
        line_data = df.iloc[i, :]
        word, definition = key_terms_generator.check_and_define_key_terms(line_data)
        
        if word is not None:
            print(f"Output {i}:")
            print("Word:", word)
            print("Definition:", definition)
        else:
            print(f"No matching complex term found in Output {i}.")

# Adjust the file path here
file_path = "projectstuff/simpletext-task2-train-input.tsv"
process_tsv_file(file_path)
