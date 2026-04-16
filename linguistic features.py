import re
import nltk
import spacy
import numpy as np

import pyphen
import stanza
import textstat
import spacy
from spacy.lang.fr.examples import sentences
spacy.cli.download("fr_core_news_sm")
import pandas as pd
from nltk.corpus import stopwords

from nltk.corpus import wordnet, stopwords, words as nltk_words
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
stanza.download('english')
stanza.download('french')
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize

def get_french_wordnet_words():
    """Get set of French words from Open Multilingual Wordnet"""
    nltk.download('omw-1.4')  # Download Open Multilingual Wordnet
    nltk.download('wordnet')  # Download WordNet

    french_words = set()
    # Get all synsets
    for synset in wordnet.all_synsets():
        # Get French lemmas for each synset
        for lemma in synset.lemmas(lang='fra'):
            # Add the French word
            french_words.add(lemma.name().lower())
    

    return set(french_words)
    # .update(nltk_stopwords.words('french')))


french_words=get_french_wordnet_words()
class TextComplexityAnalyzer:
    def __init__(self, text,language):
      self.language=language
      if language=='english':
        # Load spaCy English model
        self.nlp = spacy.load('en_core_web_sm')

        self.text = text
        self.doc = self.nlp(text)
        self.tokens = word_tokenize(text)
        self.sentences = sent_tokenize(text)

        # Precompute some analyses
        self.pos_tags = nltk.pos_tag(self.tokens)
        self.clean_tokens = [str(token) for token in self.doc if token.is_punct == False and token.is_space == False]
      elif language=='french':
        self.nlp = spacy.load('fr_core_news_sm')
        self.text = text
        self.doc = self.nlp(text)
        self.tokens = word_tokenize(text)
        self.sentences = sent_tokenize(text)

        self.pos_tags = nltk.pos_tag(self.tokens)
        self.clean_tokens = [str(token) for token in self.doc if token.is_punct == False and token.is_space == False]


    def lexical_richness(self):
        """Average lexical richness (unique words / total words)"""
        lowered_words = [word.lower() for word in self.clean_tokens]
        unique_words = len(set(lowered_words))
        return unique_words/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

    def words_before_main_verb(self):
        """Average number of words before the main verb in each sentence"""
        distances = []
        
        # Iterate through each sentence in the document
        for sent in self.doc.sents:
            # The 'root' attribute of a sentence is the token that is the root of the dependency parse tree
            root_token = sent.root
            
            # Check if the root token is a verb. It usually is, but not always.
            if root_token.pos_ == 'VERB':
                # Calculate the index of the root verb relative to the start of the sentence
                # sent[0].i is the absolute index of the first token in the sentence
                root_verb_index = root_token.i - sent[0].i
                distances.append(root_verb_index)
            else:
                main_verb_index = None
                for token in sent:
                    if token.pos_ == 'VERB':
                        main_verb_index = token.i - sent[0].i
                        break
                if main_verb_index is not None:
                    distances.append(main_verb_index)
                
        # Return the average distance
        return sum(distances) / len(distances) if distances else 0

    def entity_distance(self):
        """Calculate max distance between same entity appearances"""
        entity_positions = {}
        for ent in self.doc.ents:
            if ent.text.lower() not in entity_positions:
                entity_positions[ent.text.lower()] = []
            entity_positions[ent.text.lower()].append(ent.start)
    
        all_max_distances = []
        for positions in entity_positions.values():
            if len(positions) > 1:
                # Calculate the distance between the last and first appearance
                distance = positions[-1] - positions[0]
                all_max_distances.append(distance)
                
        # Return the maximum distance found across all entities
        return max(all_max_distances) if all_max_distances else 0

    def content_words_ratio(self):
        """Ratio of content words (nouns, verbs, adjectives, adverbs)"""
        content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
        content_words = sum(1 for token in self.doc if token.pos_ in content_pos and not token.is_punct)
        return content_words/len(self.clean_tokens)  if len(self.clean_tokens)>0 else 0

    def infrequent_words_ratio(self):
        """Ratio of words not in common English word list"""
        if self.language=='english':
          common_words = set(nltk_words.words())
          # common_words.update(nltk_stopwords.words('english'))
          infrequent_words = sum(1 for token in self.clean_tokens if token.lower() not in set(common_words))
          return infrequent_words/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0
        elif self.language=='french':
          common_words = french_words
          infrequent_words = sum(1 for token in self.clean_tokens if token.lower() not in common_words)
          return infrequent_words/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0


    def long_words_ratio(self, threshold=9):
        """Ratio of words longer than threshold"""
        long_words = sum(1 for token in self.clean_tokens if len(token) > threshold)
        return long_words/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

    def modifiers_ratio(self):
        """Ratio of modifiers (adjectives, adverbs)"""
        modifier_pos = {'ADJ', 'ADV'}
        modifiers = sum(1 for token in self.doc if token.pos_ in modifier_pos)
        return modifiers/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

    def negations_ratio(self):
        """Ratio of negation words"""
        if self.language=='english':
          negation_words = {'not', 'no', 'neither', 'nor', 'none', "never", "nothing", "nowhere", "no one",
            "can't", "don't", "won't", "isn't", "wouldn't",
            "shouldn't", "couldn't", "hadn't", "doesn't", "didn't", "haven't",
            "hasn't", "weren't", "aren't", "wasn't", "mustn't"}
        elif self.language=='french':
          negation_words={'ne',"n'", "ne", "pas", "plus", "jamais", "rien", "personne", "guère",
                        "aucun", "aucune",'non', "ni"}
        negations = sum(1 for token in self.clean_tokens if token.lower() in negation_words)
        return negations/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0


    def noun_phrases_ratio(self):
        """Ratio of noun phrases that contain consecutive nouns"""
        noun_phrase_count = len(list(self.doc.noun_chunks))


        return noun_phrase_count /len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

    def count_past_perfect_verbs(self):
        """
        Count the number of past perfect verbs in a sentence.
        """
        if self.language=='english':
          past_perfect_count = 0

          for sentence in self.sentences:
              sent_tokens = word_tokenize(sentence)
              sent_tags = nltk.pos_tag(sent_tokens)

              # Count past perfect verbs
              for i in range(len(sent_tags) - 1):
                  # Check if current word is 'had' and next word is a past participle
                  if (sent_tokens[i].lower() == 'had' and sent_tags[i+1][1] == 'VBN'):
                      past_perfect_count += 1

          # Return ratio of past perfect verbs to total tokens
          past_tense = sum(1 for _, tag in self.pos_tags if tag in ['VBD', 'VBN'])
          return past_perfect_count/past_tense if past_tense!=0 else 0
        return 0

    def verb_tense_analysis(self):
        """Analyze past tense verb ratios"""
        if self.language == 'english':
            past_tense = sum(1 for _, tag in self.pos_tags if tag in ['VBD', 'VBN'])
        else:
            past_tense = sum(1 for token in self.doc if 'Tense=Past' in token.morph)
        verbs=sum(1 for token in self.doc if token.pos_ == 'VERB')
        return past_tense/verbs if verbs!=0 else 0

    def punctuation_ratio(self):
        """Ratio of punctuation marks"""
        punctuation_count = 0
        
        for token in self.doc:
            if token.is_punct:
                punctuation_count += 1
        return punctuation_count/len(self.tokens) if len(self.tokens)>0 else 0

    def relative_clauses_ratio(self):
        """Ratio of relative clauses"""

        if self.language=='english':
          relative_pronouns = {'who', 'whom', 'whose', 'which', 'that', "when", "where", "why"}
        elif self.language=='french':
          relative_pronouns={'qui', 'que', 'quoi', 'dont', 'où', 'lequel', 'laquelle', 
                             'lesquels', 'lesquelles', 'auquel', 'à laquelle', 
                             'auxquels', 'auxquelles', 'duquel', 'de laquelle', 
                             'desquels', 'desquelles'}
        relative_clauses = sum(1 for token in self.clean_tokens if token.lower() in relative_pronouns)
        return relative_clauses/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

    def third_person_pronouns_ratio(self):
        """Ratio of third-person singular pronouns"""
        if self.language=='english':
            third_person_pronouns = {'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its',
            'they', 'them', 'their', 'theirs', 'himself', 'herself',
            'itself', 'themselves'}
        elif self.language=='french':
            third_person_pronouns = {'il', 'elle', 'on', 'ils', 'elles', 'ce', 'c',
            'le', 'la', 'les', 'l','lui', 'leur','se', 's','eux', 'soi','y', 'en',
            'celui', 'celle', 'ceux', 'celles'}
        pronouns = sum(1 for token in self.clean_tokens if token.lower() in third_person_pronouns)
        return pronouns / len(self.clean_tokens) if len(self.clean_tokens)>0 else 0


    def unique_entities_ratio(self):
        """Ratio of unique entities"""
        unique_entities = len(set(ent.text.lower() for ent in self.doc.ents))
        return unique_entities

    def readability_metrics(self):
        """Various readability metrics"""
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(self.text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(self.text)
        }

    def sentences_count_ratio(self):
        """Ratio of sentences (few sentences)"""
        return len(self.sentences)

    def words_containing_more_then_8_chars(self, threshold=8):
        """Ratio of words containing more than eight characters"""
        long_words = sum(1 for token in self.clean_tokens if len(token) > threshold)
        return long_words/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

  

    def words_per_sentence(self):
        """Average number of words per sentence"""
        words_per_sentence = [len(word_tokenize(sentence)) for sentence in self.sentences]
        return np.mean(words_per_sentence)/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0


    def consecutive_entity_distance(self):
        """Average distance between consecutive entities"""
        entity_indices = [ent.start for ent in self.doc.ents]
            
        if len(entity_indices) < 2:
            return 0
        
        # Calculate the distances between consecutive entity indices
        distances = np.diff(entity_indices)
        
        return np.mean(distances)


    def entity_metrics(self):
        """Calculate various entity-related metrics"""
        entities = list(self.doc.ents)

        # Unique entities metrics
        unique_entities = len(set(ent.text.lower() for ent in entities))

        # Entity to token ratio
        entity_token_ratio = len(entities) / len(self.clean_tokens) if len(self.clean_tokens)>0 else 0

        entity_positions = {}
        for ent in entities:
            # Use the entity's lowercase text as the key for the dictionary
            key = ent.text.lower()
            if key not in entity_positions:
                entity_positions[key] = []
            # Store the starting token index of the entity
            entity_positions[key].append(ent.start)
    
        avg_same_entity_distances = []
        for positions in entity_positions.values():
            if len(positions) > 1:
                # Calculate consecutive distances and their average
                distances = np.diff(positions)
                avg_same_entity_distances.append(np.mean(distances))
    
        return {
            'unique_entities_count': unique_entities/len(self.sentences) if len(self.sentences)>0 else 0,
            'entity_to_token_ratio': entity_token_ratio,
            'avg_same_entity_distance': np.mean(avg_same_entity_distances) if avg_same_entity_distances else 0,
            'unique_entities_to_total_num_of_entities': unique_entities/len(entities) if len(entities)>0 else 0
        }

    def clause_and_voice_analysis(self):
        """
        Analyze clauses, conjunctions, and voice
        """
        if self.language=='english':
            # Check for conditional clauses (very simplified)
            conditional_clauses = sum(1 for token in self.clean_tokens if token.lower() in {'if', 'unless', 'whether', 'in case'})
        elif self.language=='french':
            conditional_clauses = sum(1 for token in self.clean_tokens if token.lower() in {'si', 'à moins que', 'pourvu que', 'au cas où'})

        # Check for conjunctions
        conjunctions = sum(1 for token in self.doc if token.pos_ == 'CCONJ' or token.pos_ == 'SCONJ')
        if self.language =='english':
            # Check for passive voice (simplified)
            passive_voice = sum(1 for sent in self.doc.sents
                                if any(token.dep_ == 'auxpass' for token in sent))
        else:
            passive_voice = sum(1 for sent in self.doc.sents
                                if any(token.dep_ == 'aux:pass' and token.lemma_ == 'être' for token in sent))
        
        # Check for appositions
        appositions = sum(1 for token in self.doc if token.dep_ == 'appos')
        total_verbs = sum(1 for token in self.doc if token.pos_ == 'VERB')

        return {
            'conditional_clauses_ratio': conditional_clauses/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0,
            'conjunctions_ratio': conjunctions/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0,
            'passive_voice_ratio': passive_voice/total_verbs if total_verbs!=0 else 0,
            'appositions_ratio': appositions/len(self.clean_tokens) if len(self.clean_tokens)>0 else 0
        }

    def short_sentences_ratio(self, max_words=10):
        """
        Calculate the ratio of short sentences
        (Arfe et al., 2018)
        """
        # Count sentences with fewer than max_words
        short_sentences = sum(1 for sent in self.sentences
                               if len(word_tokenize(sent)) <= max_words)
        return short_sentences/len(self.sentences) if len(self.sentences)>0 else 0

    def calculate_avg_word_length(self) -> float:
        """
        Calculates the average length of words in the text using clean tokens.
        
        Returns:
            float: The mean number of characters per word. Returns 0 if no tokens exist.
        """
        # Calculate the average word length
        total_length = sum(len(word) for word in self.clean_tokens)
        return total_length / len(self.clean_tokens) if len(self.clean_tokens) > 0 else 0.0

    def syllable_to_word_ratio(self) -> float:
        """
        Calculates the average number of syllables per word.
        
        Uses the Pyphen library to hyphenate words based on the instance's 
        defined language (English or French).
        
        Returns:
            float: The ratio of total syllables to the total number of clean tokens.
        """
        if self.language == 'french':
            lang = 'fr_FR'
        elif self.language == 'english':
            lang = 'en_US'
        else:
            return 0.0
        
        # Initialize the hyphenation dictionary for the specified language
        dic = pyphen.Pyphen(lang=lang)

        # Count syllables
        total_syllables = 0
        for token in self.clean_tokens:
            # Hyphenate the word and count the syllables
            # Pyphen's .inserted() returns 'word' as 'sy-lla-ble'
            hyphenated_word = dic.inserted(token.lower())
            total_syllables += hyphenated_word.count('-') + 1

        # Calculate the ratio of syllables to words
        if len(self.clean_tokens) > 0:
            ratio = total_syllables / len(self.clean_tokens)
        else:
            ratio = 0.0  # Avoid division by zero if the text is empty

        return float(ratio)

    def get_syntactic_depth(self):
        """
        Calculates the maximum syntactic tree depth for a text.
    
        Args:
            doc (spacy.tokens.Doc): The spaCy Doc object.
    
        Returns:
            int: The maximum depth of any syntactic tree in the text.
        """
        max_depth = 0
        for token in self.doc:
            depth = 0
            current_token = token
            while current_token.head != current_token:
                current_token = current_token.head
                depth += 1
            
            # Update the maximum depth
            if depth > max_depth:
                max_depth = depth
                
        return max_depth

        

    def perform_analysis(self):
        """Aggregate all complexity metrics"""
        d= {
            'lexical_richness': self.lexical_richness(),
            'words_before_main_verb': self.words_before_main_verb(),
            'max_same_entity_distances': self.entity_distance(),
            'content_words_ratio': self.content_words_ratio(),
            'infrequent_words_ratio': self.infrequent_words_ratio(),
            'long_words_ratio': self.long_words_ratio(),
            'modifiers_ratio': self.modifiers_ratio(),
            'negations_ratio': self.negations_ratio(),
            'noun_phrases_ratio': self.noun_phrases_ratio(),
            'past_perfect_verbs': self.count_past_perfect_verbs(),
            'past_tense_verbs': self.verb_tense_analysis(),
            'punctuation_ratio': self.punctuation_ratio(),
            'relative_clauses_ratio': self.relative_clauses_ratio(),
            'sentences_number': self.sentences_count_ratio(),
            'third_person_pronouns_ratio': self.third_person_pronouns_ratio(),
            'unique_entities': self.unique_entities_ratio(),
            'words_containing_more_then_8_characters': self.words_containing_more_then_8_chars(),
            'words_per_sentence': self.words_per_sentence(),
            'consecutive_entity_distance': self.consecutive_entity_distance(),
            'flesch_reading_ease': self.readability_metrics()['flesch_reading_ease'],
            'unique_entities_average': self.entity_metrics()['unique_entities_count'],
            'avg_same_entity_distance': self.entity_metrics()['avg_same_entity_distance'],
            'entity_to_token_ratio': self.entity_metrics()['entity_to_token_ratio'],
            'flesch_kincaid_grade': self.readability_metrics()['flesch_kincaid_grade'],
            'unique_entities_to_total_num_of_entities': self.entity_metrics()['unique_entities_to_total_num_of_entities'],
            'appositions_ratio': self.clause_and_voice_analysis()['appositions_ratio'],
            'conditional_clauses_ratio': self.clause_and_voice_analysis()['conditional_clauses_ratio'],
            'conjunctions_ratio': self.clause_and_voice_analysis()['conjunctions_ratio'],
            'passive_voice_ratio': self.clause_and_voice_analysis()['passive_voice_ratio'],
            'short_sentences_ratio': self.short_sentences_ratio(),
            'syntactic_tree_depth': self.get_syntactic_depth(),
            'syllables_ratio': self.syllable_to_word_ratio(),
            'avg_word_length': self.calculate_avg_word_length()

            #'concreteness': self.concreteness_analysis()
        }
        if self.language=='french':
          del d['past_perfect_verbs']
        return d

def add_text_complexity_metrics(df, text_column, language):
    """
    Calculates linguistic complexity metrics for a given text column.
    Handles columns with single strings and lists of strings.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing the text data.
        language (str): The language of the text ('english' or 'french').

    Returns:
        pd.DataFrame: A new DataFrame with the calculated metrics as new columns.
    """
    df_copy = df.copy()

    def process_text_or_list(text_data):
        if isinstance(text_data, list):
            if not text_data:
                # Return a series of zeros if the list is empty
                return pd.Series([0] * len(TextComplexityAnalyzer("sample", language).perform_analysis().keys()), 
                                 index=TextComplexityAnalyzer("sample", language).perform_analysis().keys())
            
            # Calculate metrics for each string in the list
            list_of_metrics_series = [
                TextComplexityAnalyzer(str(item), language).perform_analysis()
                for item in text_data
            ]

            # Convert to DataFrame and calculate the mean for each metric
            metrics_df = pd.DataFrame(list_of_metrics_series)
            return metrics_df.mean()

        else:
            # Process a single string as before
            return pd.Series(TextComplexityAnalyzer(str(text_data), language).perform_analysis())

    # Apply the new processing function to the text column
    complexity_metrics = df_copy[text_column].apply(process_text_or_list)

    # Add metrics as new columns
    df_copy = pd.concat([df_copy, complexity_metrics], axis=1)

    return df_copy


files=[
    'Clear','WikiLarge FR', 
    'asset', 
    'MultiCochrane',
    'WikiAuto', 
       ]


for input_file in files:
    xls_file = pd.ExcelFile(f'llm output/{input_file} output.xlsx')
    df = pd.read_excel(xls_file)
    llm_dfs = {}
    for sheet in xls_file.sheet_names:
        df = pd.read_excel(xls_file, sheet_name=sheet)
        if input_file in ['Clear','WikiLarge FR']:
            language = 'english'
        elif input_file in ['WikiAuto', 'asset', 'MultiCochrane']:
            language = 'french'
        df_with_metrics = add_text_complexity_metrics(df[[sheet]],f'{sheet} response',language)
        llm_dfs[sheet] = df_with_metrics
        
    output_file=f'bats features output/{input_file} with bats metrics.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        # Loop through the dictionary and create a sheet for each key
        for sheet_name, sheet_data in llm_dfs.items():
            # Convert the sheet data to a DataFrame
            df = pd.DataFrame(sheet_data)
            df.to_excel(writer, sheet_name=sheet_name, index=False)