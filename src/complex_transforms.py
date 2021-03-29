from src.BART_models import BART_Query_ReWriter, BART_Simple
from src.text_transforms import *
from src.useful_utils import chunks

from tqdm import tqdm 
import torch
from itertools import permutations 
import random
from scipy.interpolate import interp1d

    
class BART_Query_Rewriter_Transform():
    def __init__(self, checkpoint_path, device=None, no_tqdm=False, **kwargs):
        '''
        A Transform that re-writes unresolve queries based on previous turns.
        
        checkpoint_path: str: path to only the **state dict** of the model, loaded with load_state_dict
        '''
        if device:
            self.device = device
        else:
            self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.no_tqdm = no_tqdm
        self.BART_query_rewriter = BART_Query_ReWriter(**kwargs)
        self.BART_query_rewriter.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.BART_query_rewriter.to(self.device)
        self.BART_numericalise_transform = Numericalise_Transform(fields=[('input_text','input_ids')])
        self.BART_denumericalise_transform = Denumericalise_Transform(fields=[('pred_ids','rewritten_query')])
        self.rewriter_context_query_merge_transform = Rewriter_Context_Query_Merge_Transform()
        print(f"BERT ReRanker initialised on {self.device}. Batch size {1}")
    
    def __call__(self, samples, **kwargs):
        '''
        samples: [dict]: [{'unresolved_query':'unresolved query text', 'previous_queries':['first query text', 'second query text']}]
        returns: [dict]: [{'rewritten_query':'query text', 'unresolved_query':'unresolved query text', 'previous_queries':['first query text',]}]
        '''
        samples = self.rewriter_context_query_merge_transform(samples)
        samples = self.BART_numericalise_transform(samples)
        if self.no_tqdm:
            pbar = samples
        else:
            pbar = tqdm(samples, desc="Re-Writing queries")
        for sample_obj in pbar:
            input_ids = sample_obj["input_ids"]
            
            output_ids = self.BART_query_rewriter.generate(torch.tensor([input_ids], device=self.device), num_beams=4, max_length=512, early_stopping=True)
            single_out_ids = output_ids[0].tolist()
            sample_obj["pred_ids"] = single_out_ids
        samples = self.BART_denumericalise_transform(samples)
        return samples

class BART_Full_Conversational_Rewriter_Transform():
    def __init__(self, checkpoint_path, **kwargs):
        '''
        This Transform takes a sequence of raw queries and re-writes them to the resolved version fed off itself.
        '''
        self.BART_query_rewriter_transform = BART_Query_Rewriter_Transform(checkpoint_path, no_tqdm=True, **kwargs)
        self.query_cleaner_transform = Query_Cleaner_Transform(fields=[('rewritten_query','cleaned_rewritten_query')])
        self.cached_generations = {}
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'unresolved_query':"third raw query", 'previous_queries':['first raw query', 'second raw query']}]
        returns: [dict]: [{'rewritten_query':"third raw query", 'full_rewritten_queries':['first rewritten q', 'second rewritten q', 'thir..'],...]
        '''
        for sample_obj in tqdm(samples, desc="BART self feeding rewrites"):
            raw_queries = sample_obj['previous_queries'] + [sample_obj['unresolved_query']]
            rewritten_queries = []
            for i, raw_query in enumerate(raw_queries):
                if tuple(raw_queries[:i+1]) in self.cached_generations:
                    rewritten_query = self.cached_generations[tuple(raw_queries[:i+1])]
                else:
                    rewritten_samples = self.BART_query_rewriter_transform([{'unresolved_query':raw_query, 'previous_queries':rewritten_queries}])
                    rewritten_sample = self.query_cleaner_transform(rewritten_samples)[0]
                    rewritten_query = rewritten_sample['cleaned_rewritten_query']
                    self.cached_generations[tuple(raw_queries[:i+1])] = rewritten_query
                
                rewritten_queries.append(rewritten_query)
            sample_obj['full_rewritten_queries'] = rewritten_queries
            sample_obj['rewritten_query'] = rewritten_queries[-1]
        return samples
    
class BART_Conditional_Generator_Transform():
    def __init__(self, model_or_path, device=None, show_tqdm=True, numericaliser="BART", denumericaliser='BART', config=None, chunk_size=64, pad_id=1, **kwargs):
        '''
        A Transform that generates a token sequence given another sequence. It uses the BART tokenizer for input and output.
        
        model_or_path: str or pytorch module: path to a Pytorch Lightning checkpoint: {'state_dict':...} or a model class
        '''
        self.chunk_size = chunk_size
        if device:
            self.device = device
        else:
            if isinstance(model_or_path, str):
                self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = model_or_path.device
            print(f"Running model on {self.device}")
        self.show_tqdm = show_tqdm
        if isinstance(model_or_path, str):
            ckpt = torch.load(model_or_path, map_location=self.device)
            self.BART_conditional_generator = BART_Simple(config=config,**kwargs)
            self.BART_conditional_generator.load_state_dict(ckpt['state_dict'])
        else:
            self.BART_conditional_generator = model_or_path
        self.BART_conditional_generator.to(self.device)
        self.BART_conditional_generator.eval()
        self.BART_numericalise_transform = Numericalise_Transform(numericaliser=numericaliser, fields=[('input_text','input_ids')], **kwargs)
        self.PAD = pad_id
        self.BART_denumericalise_transform = Denumericalise_Transform(denumericaliser=denumericaliser, fields=[('pred_ids','pred_text')], **kwargs)
    
    def __delete__(self, instance):
        del self.BART_conditional_generator
        torch.cuda.empty_cache()
        
    def __call__(self, samples):
        '''
        samples: [dict]: [{'input_text':'text to condition on'}]
        returns: [dict]: [{'input_text':'text to condition on', 'pred_text':"text from BARTs decoder"}]
        '''
        samples = self.BART_numericalise_transform(samples)
        if self.show_tqdm:
            pbar = tqdm(list(chunks(samples,self.chunk_size)), desc="BART is thinking:")
        else:
            pbar = samples
        for chunk in pbar:
            input_tensor = torch.nn.utils.rnn.pad_sequence(
                                [torch.tensor(sample_obj["input_ids"], dtype=torch.long) for sample_obj in chunk], 
                                                     padding_value=self.PAD).T.to(self.device)
            attention_mask = (input_tensor != self.PAD).type(torch.float).to(self.device)
            output_ids = self.BART_conditional_generator.generate(input_tensor, attention_mask=attention_mask, pad_token_id=self.PAD, num_beams=4, max_length=512, early_stopping=False)
            for i in range(len(chunk)):
                single_out_ids = output_ids[i].tolist()
                chunk[i]["pred_ids"] = single_out_ids
            del input_tensor
            del attention_mask
            del output_ids
        samples = self.BART_denumericalise_transform(samples)
        return samples