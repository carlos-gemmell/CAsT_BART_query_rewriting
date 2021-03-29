import copy
from transformers import BertTokenizer, BartTokenizer
from tokenizers import processors, Tokenizer
from tqdm import tqdm 
import random
import ujson
import re 
import os
import pickle
                
    
class Numericalise_Transform():
    def __init__(self, numericaliser='BART', fields=[("input_text","input_ids")], debug=True, max_len=1000, **kwargs):
        if numericaliser == 'BART':
            self.numericaliser = BartTokenizer.from_pretrained('facebook/bart-large').encode
        elif numericaliser == 'BERT':
            self.numericaliser = BertTokenizer.from_pretrained('bert-base-uncased').encode
        else:
            self.numericaliser = numericaliser
        if debug:
            print(f"Numericaliser. Ex: 'This is a test' -> {self.numericaliser('This is a test')}")
        self.fields = fields
        self.max_len = max_len
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'input_text':"text and more", ...}]
        returns: [dict]: [{'input_ids':[34,2,8...], 'input_text':"text and more", ...}]
        '''
        for sample_obj in samples:
            for str_field, id_field in self.fields:
                sample_obj[id_field] = self.numericaliser(sample_obj[str_field])[:self.max_len]
        return samples
    
    
class Denumericalise_Transform():
    def __init__(self, denumericaliser='BART', fields=[("input_ids","input_text")], debug=True, skip_special_tokens=True, **kwargs):
        if denumericaliser == 'BART':
            self.denumericaliser = BartTokenizer.from_pretrained('facebook/bart-large').decode
        elif denumericaliser == 'BERT':
            self.denumericaliser = BertTokenizer.from_pretrained('bert-base-uncased').decode
        else:
            self.denumericaliser = denumericaliser
        if debug:
            print(f"Denumericaliser. Ex: [0,1,2,3,4,5,6,7,8,9] -> {self.denumericaliser([0,1,2,3,4,5,6,7,8,9])}")
        self.fields = fields
        self.skip_special_tokens = skip_special_tokens
    
    def __call__(self, samples):
        '''
        sample_obj: [dict]: [{'input_ids':[34,2,8...],...}]
        returns: [dict]: [{'input_ids':[34,2,8...], 'input_text':"text and more", ...}]
        '''
        for sample_obj in samples:
            for str_field, id_field in self.fields:
                sample_obj[id_field] = self.denumericaliser(sample_obj[str_field], skip_special_tokens=self.skip_special_tokens)
        return samples
    

    
class Rewriter_Query_Resolver_Transform():
    def __init__(self, get_query_fn, prev_queries_utter_type="manual_rewritten_utterance", fields={}, **kwargs):
        '''
        get_query_fn: fn(q_id) -> "query string"
        '''
        self.fields = {
            'q_id_field':'q_id',
            'prev_turns_field':'prev_turns',
            'unresolved_query_field':'unresolved_query',
            'previous_queries_field':'previous_queries',
            'resolved_query_field':'resolved_query'
        }
        self.fields.update(fields)
        self.get_query_fn = get_query_fn
        self.prev_queries_utter_type = prev_queries_utter_type
        
    def __call__(self, samples):
        '''
        samples [dict]: [{'q_id':"32_4", 'prev_turns':["32_3",..]},...]
        returns: [dict]: [{'unresolved_query':'query text', 'resolved_query':'query text', 'previous_queries':['first query text', 'second query text']}]
        '''
        for sample_obj in samples:
            sample_obj[self.fields["unresolved_query_field"]] = self.get_query_fn(sample_obj[self.fields["q_id_field"]], utterance_type='raw_utterance')
            previous_queries = [self.get_query_fn(q_id, utterance_type=self.prev_queries_utter_type)for q_id in sample_obj[self.fields["prev_turns_field"]]]
            sample_obj[self.fields["previous_queries_field"]] = previous_queries
            sample_obj[self.fields["resolved_query_field"]] = self.get_query_fn(sample_obj[self.fields["q_id_field"]], utterance_type='manual_rewritten_utterance')
        return samples
    
class Rewriter_Context_Query_Merge_Transform():
    def __init__(self, **kwargs):
        '''
        This Transform merges queries from previous turns and the current unresolved query into a single input sequence.
        '''
        pass
    
    def __call__(self, samples):
        '''
        samples: [dict]: [{'unresolved_query':'query text', 'previous_queries':['first query text', 'second query text']}]
        returns: [dict]: [{'input_text':'merged query text', 'unresolved_query':'query text', 'previous_queries':['first query text',]}]
        '''
        for sample_obj in samples:
            sample_obj["input_text"] = " ".join(sample_obj['previous_queries']) + " query: " + sample_obj['unresolved_query']
        return samples
    
class Rewriter_Context_Target_Transform():
    def __init__(self,  merge_mode="full_context_rewrite", **kwargs):
        '''
        This Transform merges queries from previous turns and the current RESOLVED target query to make the target sequence to be predicted.
        '''
        self.merge_mode = merge_mode
    
    def __call__(self, samples):
        '''
        samples: [dict]: [{'resolved_query':'resolved query text', 'previous_queries':['first query text', 'second query text']}]
        returns: [dict]: [{'target_text':'merged query text', 'unresolved_query':'query text', 'previous_queries':['first query text',]}]
        '''
        for sample_obj in samples:
            if self.merge_mode == "full_context_rewrite":
                sample_obj["target_text"] = " ".join(sample_obj['previous_queries']) + " query: " + sample_obj['resolved_query']
            elif self.merge_mode == "last_turn_rewrite":
                sample_obj["target_text"] = sample_obj['resolved_query']
        return samples
    
class Query_Cleaner_Transform():
    def __init__(self, fields=[('query','cleaned_query')]):
        '''
        This Transform removes some of  the un-necessary halucinated text from the query re-writer.
        '''
        self.fields = fields
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'query':"query: query text? what else?!", 'unresolved_query':'query text?'}]
        returns: [dict]: [{'cleaned_query':"query text?", 'query':"query: query text? what else?!"}]
        '''
        for sample_obj in samples:
            for input_field, target_field in self.fields:
                query = sample_obj[input_field]
                new_query = query.split("query:")[-1]
                if "?" in new_query:
                    new_query = new_query[:new_query.index('?')+1]
                # keep the same amount of sentences as the unresolved query
                split_unres_query = re.sub(r'(\?|\.)', '\\1[cut]',  sample_obj['unresolved_query'][:])
                unres_query_sents = list(filter(None, split_unres_query.split('[cut]')))
                num_unres_query_sents = len(unres_query_sents)
                
                split_query = re.sub(r'(\?|\.)', '\\1[cut]',  new_query[:])
                unres_query_sents = list(filter(None, split_query.split('[cut]')))
                new_query = ''.join(unres_query_sents[:num_unres_query_sents])
                
                sample_obj[target_field] = new_query
        return samples
        


class Rename_Transform():
    def __init__(self, fields=[]):  
        '''
        This is a stateless function transform that renames fields already existing in a sequence of samples to new names.
        Fields can be set to None to delete them.
        fields: [('fieldA', 'fieldB')]
        '''
        self.fields = fields
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'fieldA': 'foo bar'}...]
        retrns: [dict]: [{'fieldB': 'foo bar'}...]
        '''
        for sample_obj in samples:
            for src_field, tgt_field in self.fields:
                if tgt_field == None:
                    sample_obj.pop(src_field)
                else:
                    sample_obj[tgt_field] = sample_obj.pop(src_field)
        return samples
    
    
    