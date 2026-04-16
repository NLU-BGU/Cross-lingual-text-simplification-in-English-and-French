import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
# Load variables from .env into environment variables
load_dotenv()

# Initialize the client 
# It will automatically find the key from os.environ["OPENAI_API_KEY"]
client = OpenAI()

SYSTEM_PROMPT = (
    "You are a text-to-text model. Your sole purpose is to provide the final output of a requested task. "
    "Do not include any interim steps, intermediate results, or conversational filler. "
    "Your response must begin directly with the final, complete answer."
)

def get_completion(prompt, model="gpt-4o"):
    """Helper function to streamline API calls"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3 # Lower temperature for more consistent, direct outputs
    )
    return response.choices[0].message.content

def direct_prompt(txt, model, lang='French'):
    prompt = f"Please simplify the following text in {lang}: {txt}"
    return get_completion(prompt, model)

def cot_translate_then_simplify_prompt(txt, model, lang='French'):
    prompt = f"Please first translate the following text to {lang} and then simplify the translated text in {lang}: {txt}"
    return get_completion(prompt, model)

def cot_simplify_then_translate_prompt(txt, model, lang='French'):
    prompt = f"Please first simplify the following text and then translate the simplification to {lang}: {txt}"
    return get_completion(prompt, model)

def pipeline_translate_then_simplify_prompt(txt, model, lang='French'):
    # Step 1: Translate
    prompt1 = f"Please translate the following text to {lang}: {txt}"
    translated = get_completion(prompt1, model)
    
    # Step 2: Simplify
    prompt2 = f"Please simplify the following text in {lang}: {translated}"
    return get_completion(prompt2, model)

def pipeline_simplify_then_translate_prompt(txt, model, lang='French'):
    # Step 1: Simplify
    prompt1 = f"Please simplify the following text: {txt}"
    simplified = get_completion(prompt1, model)
    
    # Step 2: Translate
    prompt2 = f"Please translate the following text to {lang}: {simplified}"
    return get_completion(prompt2, model)



propmt_types = ['direct', 'CoT translate->simplify', 'CoT simplify->translate', 'pipeline translate->simplify',
               'pipeline simplify->translate']

for file in os.listdir('input'):
    if file.endswith('.xlsx'):
        os.mkdir(f'output {model}')
        results = {key: [] for key in propmt_types}
        file_name = file.split('.')[0]
        df = pd.read_excel(f'input/{file}')
        complex_text = [col for col in list(df.columns) if 'Complex' in col][0]
        complex_lang = complex_text.split(' ')[0]
        if complex_lang=='English':
            lang='French'
        elif complex_lang=='French':
            lang='English'
        print(f"{file_name} {complex_text} {lang}")
        complex_text = list(df[complex_text])
        for i, txt in enumerate(complex_text):
    
            direct = direct_prompt(txt, model, lang)
            print(f'direct: \n{direct}\n')
            cot_translate_simplify = cot_translate_then_simplify_prompt(txt, model, lang)
            print(f'CoT translate then simplify: \n{cot_translate_simplify}\n')
            cot_simplify_translate = cot_simplify_then_translate_prompt(txt, model, lang)
            print(f'CoT simplify then translate: \n{cot_simplify_translate}\n')
            pipeline_translate_simplify = pipeline_translate_then_simplify_prompt(txt, model, lang)
            print(f'pipeline translate then simplify: \n{pipeline_translate_simplify}\n')
            pipeline_simplify_translate = pipeline_simplify_then_translate_prompt(txt, model, lang)
            print(f'pipeline simplify then translate: \n{pipeline_simplify_translate}')       

            

            results['direct'].append(direct)
            results['CoT translate->simplify'].append(cot_translate_simplify)
            results['CoT simplify->translate'].append(cot_simplify_translate)
            results['pipeline translate->simplify'].append(pipeline_translate_simplify)
            results['pipeline simplify->translate'].append(pipeline_simplify_translate)
        
        with pd.ExcelWriter(f'output {model}/{file_name} {model} response.xlsx', engine='openpyxl') as writer:
            for sheet_name, content in results.items():
                # Convert the list/dict into a Pandas DataFrame
                df = pd.DataFrame(content)
                
                # Write the DataFrame to a specific sheet
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
        