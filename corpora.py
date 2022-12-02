import pandas as pd
import numpy as np
import os
from os.path import join
import validators

from model_test import process_answer

CWD = os.getcwd()

def create_corpus(files, name):

    with open(join(CWD, 'corpora', name), 'w', encoding="utf-8") as output:
        
        for f in files:
            text = ""
            
            df = pd.read_json(join(CWD, 'data', f))

            for doc_id, answer in enumerate(df["answer"]):

                
                if validators.url(answer):
                    continue

                sentences = process_answer(answer)
                if len(sentences) == 0:
                    continue
                
                for sentence in sentences:
            
                    for term in sentence:
                        text += f"{term} "
                    
                    text = text[:-1] + "."
                
                if doc_id == 9:
                    print(sentences)
                    print(len(answer))

                text += "\n"
            
            output.write(text)


def mutate_corupus(name)

create_corpus([
    "q_2_k_12.json",
    "q_2_k_15.json",
    "q_2_k_18.json",
    "q_2_k_20.json",
    "q_2_k_24.json",
    "q_2_k_30.json",
    "q_2_k_32.json",
    "q_2_k_36.json",
], "all.txt")