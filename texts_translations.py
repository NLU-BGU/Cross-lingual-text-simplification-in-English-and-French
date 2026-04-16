from deep_translator import GoogleTranslator
import pandas as pd

def translate_text(text, source_lang, target_lang):
    """
    Translates text using the Google Translate engine.
    """
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    return translator.translate(text)


files=[
    'Clear','WikiLarge FR', 
    'asset', 
    'MultiCochrane',
    'WikiAuto', 
       ]


for file in files:
    df = pd.read_excel(f'input/{file}.xlsx')
    if file in ['Clear', 'WikiLarge FR']:
        df['English Translated'] = df.apply(lambda row: translate_text(row['French Complex'].tolist(), 'fr', 'en'), axis=1)
        df['English Simplified'] = df.apply(lambda row: translate_text(row['French Simplified'].tolist(), 'fr', 'en'), axis=1)
    elif file in ['asset', 'WikiAuto', 'MultiCochrane']:
        
        df['French Translated'] = df.apply(lambda row: translate_text(row['English Complex'].tolist(), 'en', 'fr'), axis=1)
        if file!='MultiCochrane':
            df['French Simplified'] = df.apply(lambda row: translate_text(row['English Simplified'].tolist(), 'en', 'fr'), axis=1)
    df.to_excel(f'output/{file} translated.xlsx', index=False)