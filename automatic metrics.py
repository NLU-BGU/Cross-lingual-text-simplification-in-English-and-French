import pandas as pd
import easse.easse
from bert_score import score
from easse.easse import bleu,sari
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util
import torch

def calculate_sari(reference, prediction, originals):

    refs_sents = [[str(reference)]]
    orig_sents=[originals]
    sys_sents=[prediction]
    return sari.corpus_sari(refs_sents=refs_sents, sys_sents=sys_sents, orig_sents=orig_sents)

def calculate_bleu(reference, prediction):
    
    sys_sents = [prediction]
    refs_sents = [[reference]]
    return bleu.corpus_bleu(refs_sents=refs_sents, sys_sents=sys_sents)

scorer = BERTScorer(lang="en", rescale_with_baseline=True)

# Load a Sentence-CamemBERT model
model = SentenceTransformer('dangvantuan/sentence-camembert-base')

def calculate_camembert_similarity(reference, prediction):
    
    
    # Encode sentences to get their embeddings
    sentence_embedding = model.encode(prediction, convert_to_tensor=True)
    references_embeddings = model.encode(reference, convert_to_tensor=True)

    # Calculate the cosine similarity between your sentence and all references
    cosine_scores = util.pytorch_cos_sim(sentence_embedding, references_embeddings)
    
    average_score = torch.mean(cosine_scores).item()

    return average_score

files=[
    'Clear','WikiLarge FR', 
    'asset', 
    'MultiCochrane',
    'WikiAuto', 
       ]

for input_file in files:

    xls_file = pd.ExcelFile(f'llm output/{input_file} output.xlsx')
    llm_dfs = {}
    
    for sheet in xls_file.sheet_names:
        df = pd.read_excel(xls_file, sheet_name = sheet)
        prediction = [col for col in df.columns if 'response' in col][0]
        if input_file in ['Clear','WikiLarge FR']:
            reference = 'English Simplified'
            complex = 'English Translated'
            language = 'english'
        elif input_file in ['WikiAuto', 'asset', 'MultiCochrane']:
            reference = 'French Simplified'
            complex = 'French Translated'
            language = 'french'
        
        df['bleu_score'] = df.apply(lambda row: calculate_bleu(row[reference], row[prediction]), axis=1)
        df['sari_score'] = df.apply(lambda row: calculate_sari(row[reference], row[prediction], row[complex]), axis=1)

        if language == 'english':
            P, R, F1 = scorer.score(cands = df[prediction].tolist(), refs = df[reference].tolist())
            df['bert_score'] = F1
        elif language == 'french':
            df['camembert_score'] = df.apply(lambda row: calculate_camembert_similarity(row[reference], row[prediction]), axis=1)

        llm_dfs[sheet] = df

    with pd.ExcelWriter(f'automatic metrics results/{input_file} with automatic metrics.xlsx', engine='xlsxwriter') as writer:
            # Loop through the dictionary items (sheet_name, data)
            for sheet_name, sheet_data in llm_dfs.items():
                # Convert the sheet data dictionary to a DataFrame
                df = pd.DataFrame(sheet_data)
                
                # Write the DataFrame to the specific sheet
                df.to_excel(writer, sheet_name=sheet_name, index=False)