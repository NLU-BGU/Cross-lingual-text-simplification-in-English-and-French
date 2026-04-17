import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import argparse
from typing import Optional
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

def get_completion(prompt, model="gpt-4o-mini"):
    """Helper function to streamline API calls"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
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



PROMPT_TYPES = [
    'direct',
    'CoT translate->simplify',
    'CoT simplify->translate',
    'pipeline translate->simplify',
    'pipeline simplify->translate',
]


def _infer_target_language(complex_column_name: str) -> str:
    complex_lang = complex_column_name.split(' ')[0]
    if complex_lang == 'English':
        return 'French'
    if complex_lang == 'French':
        return 'English'
    return 'French'


def run(input_dir: str, model: str, output_dir: Optional[str] = None) -> None:
    out_dir = output_dir or f"output {model}"
    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.xlsx'):
            continue

        results = {key: [] for key in PROMPT_TYPES}
        file_stem = os.path.splitext(filename)[0]

        df_in = pd.read_excel(os.path.join(input_dir, filename))
        complex_candidates = [col for col in df_in.columns if 'Complex' in str(col)]
        if not complex_candidates:
            raise ValueError(f"No column containing 'Complex' found in {filename}")
        complex_col = str(complex_candidates[0])
        lang = _infer_target_language(complex_col)

        for txt in df_in[complex_col].tolist():
            results['direct'].append(direct_prompt(txt, model, lang))
            results['CoT translate->simplify'].append(cot_translate_then_simplify_prompt(txt, model, lang))
            results['CoT simplify->translate'].append(cot_simplify_then_translate_prompt(txt, model, lang))
            results['pipeline translate->simplify'].append(pipeline_translate_then_simplify_prompt(txt, model, lang))
            results['pipeline simplify->translate'].append(pipeline_simplify_then_translate_prompt(txt, model, lang))

        out_path = os.path.join(out_dir, f"{file_stem} {model} response.xlsx")
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            for sheet_name, content in results.items():
                pd.DataFrame(content).to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prompting strategies over input XLSX files.")
    parser.add_argument("--input-dir", default="input", help="Directory containing input .xlsx files.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name.")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: 'output <model>').")
    args = parser.parse_args()

    run(input_dir=args.input_dir, model=args.model, output_dir=args.output_dir)


if __name__ == "__main__":
    main()