from datetime import datetime
import re
import random
import os
import gzip
import json
import requests
import urllib
import jsonlines
import numpy as np
from  heapq import heappush, heappop, nsmallest
from anytree import Node, RenderTree, NodeMixin, find_by_attr
from tqdm import tqdm  
from pytorch_lightning import Callback
            
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    file_size = int(urllib.request.urlopen(url).info().get('Content-Length', -1))
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size
    


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

            

def dummy_scorer(run):
    '''
    run: [(query, doc)]
    '''
    return list(np.random.uniform(low=0.0, high=1.0, size=(len(run),)))



def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k.
    
    >>> with open('really_big_file.dat') as f:
    >>> for piece in read_in_chunks(f):
    >>>     process_data(piece)
    """
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data
        
def normalize(x):
    return x/np.sum(x)


