import os
import json
import string
import torch
import nltk
import faiss
import numpy as np
from tqdm import tqdm

from langdetect import detect as detect_lang
from typing import Dict, List



class Solution:
    def __init__(self):
        self.EMB_PATH_KNRM = os.environ['EMB_PATH_KNRM']
        self.VOCAB_PATH = os.environ['VOCAB_PATH']
        self.MLP_PATH = os.environ['MLP_PATH']
        self.EMB_PATH_GLOVE = os.environ['EMB_PATH_GLOVE']
        
        self.glove_emb = None
        self.knrm_emb = None
        self.knrm_vocab = None
        self.initial_data_loaded = True
        self.question_loaded = False
        self.raw_documents = {}
        self.documents = {}
        self.emb_documents ={}
        self.faiss_index = faiss.IndexFlatL2(50)
        

    def filter_query(self, query: str):
        return detect_lang(query) == 'en'

    def get_suggestions(self, query: str):
        emb_query = self.get_query_emb(query)
        
        D, I = self.faiss_index.search(np.array([emb_query]), 10)
        
        rer = [(str(idx), self.raw_documents[str(idx)]) for idx in I[0] if idx!=-1]
        return rer
    
    def get_query_emb(self, query):        
        d_proc = query.lower().strip()
        for p in string.punctuation:
            d_proc = d_proc.replace(p, ' ').strip()
        d_proc = nltk.word_tokenize(d_proc)
        embs = []
        for token in d_proc:
            knrm_vocab_idx = self.knrm_vocab.get(token, None)
            glove_emb_flag = self.glove_emb.get(token, None)
            if knrm_vocab_idx and glove_emb_flag:
                token_emb = self.knrm_emb[knrm_vocab_idx].numpy()
                embs.append(token_emb)
            else:
                d_emb = np.zeros((50,))
        d_emb = np.mean(embs, axis=0)
        return d_emb
            

    def read_glove_embeddings(self) -> Dict[str, List[str]]:
        with open(self.EMB_PATH_GLOVE, 'r') as rfile:
            raw_data = rfile.readlines()

        emb_dict = {}
        for line in raw_data:
            l = line.replace('\n', '').split(' ')
            emb = [float(i) for i in l[1:]]
            emb_dict[l[0]] = emb
        return emb_dict

    def read_knrm_embeddings(self):
        return torch.load(self.EMB_PATH_KNRM)['weight']

    def read_knrm_vocab(self):
        with open(self.VOCAB_PATH, 'rb') as rfile:
            vocab = json.load(rfile)
        return vocab

    def read_mlp(self):
        return torch.load(self.MLP_PATH)

    def preprocess_documents(self, docs):
        proc_docs = []
        for idx, d in tqdm(docs.items()):
            d_proc = d.strip().lower()
            for p in string.punctuation:
                d_proc = d_proc.replace(p, ' ').strip()
            d_proc = nltk.word_tokenize(d_proc)
            proc_docs.append(d_proc)
            self.raw_documents[idx] = d
            self.documents[idx] = d_proc

    def create_index(self):
        docs_embeddings = []
        idxs = []
        for idx, d in tqdm(self.documents.items()):
            embs = []
            for token in d:
                if self.glove_emb.get(token, None):
                    token_emb = self.knrm_emb[self.knrm_vocab[token]].numpy()
                    embs.append(token_emb)
            if len(embs) == 0:
                d_emb = np.zeros((50,))
            else:
                d_emb = np.mean(embs, axis=0)
            docs_embeddings.append(d_emb)
            self.emb_documents[idx] = d_emb
            idxs.append(idx)
        docs_embeddings = np.asarray(docs_embeddings, dtype=np.float32)
        idxs = np.asarray(idxs, dtype=np.int64)
        # self.faiss_index.add_with_ids(docs_embeddings, idxs)
        
        self.faiss_index = faiss.IndexIDMap(self.faiss_index)
        self.faiss_index.add_with_ids(docs_embeddings, idxs)
    
    
    def read_data(self):
        self.glove_emb = self.read_glove_embeddings()
        self.knrm_emb = self.read_knrm_embeddings()
        self.knrm_vocab = self.read_knrm_vocab()
        self.initial_data_loaded = True