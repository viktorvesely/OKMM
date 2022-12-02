import pandas as pd
import numpy as np
import os
from os.path import join
import re
import time

CWD = os.getcwd()


def retrain(model_name, new_name, corpus_file):
    import gensim.models.fasttext as gFasttext
    from gensim.test.utils import datapath

    path = datapath(join(os.getcwd(), 'models', model_name))
    print("Loading the old model")
    model = gFasttext.load_facebook_model(path)

    corpus_path = join(CWD, 'corpora', corpus_file)
    print("Building vocab")
    model.build_vocab(corpus_file=corpus_path, update=True)

    print("Retraining")
    model.train(
        corpus_file=corpus_path,
        epochs=10,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words
    )

    print("Saving")
    model.save(join(CWD, 'models', new_name))

def create_corpus(df, name):
    
    path = join(CWD, 'corpora', name)
    
    corpus = ""

    for doc_id, answer in enumerate(df["answer"]):

        sentences = process_answer(answer)
        if len(sentences) == 0:
            continue
        
        for sentence in sentences:
     
            for term in sentence:
                corpus += f"{term} "
            
            corpus = corpus[:-1] + "."
        
        if doc_id == 9:
            print(sentences)
            print(len(answer))

        corpus += "\n"


    with open(path, "w") as f:
        f.write(corpus)
    

def load_model(modelname):
    import fasttext
    return fasttext.load_model(join(os.getcwd(), 'models', modelname))

def load_gensim(modelname):
    from gensim.models import fasttext
    from gensim.test.utils import datapath

    path = datapath(join(CWD, "models", modelname))
    return fasttext.load_facebook_vectors(path)
    

def load_data(name):
    return pd.read_json(join(os.getcwd(), 'data', name))

def process_answer(answer):
    answer = re.sub('[^a-zA-Z0-9 \nÁ-ž\.]', '', answer)
    sentences = answer.split(".")

    terms = []
    for sentence in sentences:
        preterms = sentence.split(" ")
        if all('' == term for term in preterms):
            continue
        terms.append(preterms)

    return terms

def process_nearest_terms(df, model, new_name):
    n_documents = df["answer"].size

    print("Processing terms")
    proc_df = df.copy()
    for doc_id, answer in enumerate(df["answer"]):

        nearest_answer = ""
        for sentence in process_answer(answer):
            for term in sentence:
                nearest_term = model.get_nearest_neighbors(term, k=1)[0][1]
                nearest_answer += f"{nearest_term} "
            nearest_answer = nearest_answer[:-1] + "."

        proc_df.loc[doc_id, "answer"] = nearest_answer  

        print(f"{doc_id+1} / {n_documents}", end='\r')  
    
    print()
    proc_df.to_csv(new_name)


def time_word_to_vec(model, word):
    start = time.perf_counter()
    for _ in range(100):
        temp = model[word]
    end = time.perf_counter()

    print(f"W2v: {(end - start) / 100}")

def create_answer_vectors(model, df, path):

    n_docs = df["answer"].size
    answers = np.zeros((n_docs, 300))

    for document_id, answer in enumerate(df["answer"]):            

        answer_vector = []
            
        sentences = process_answer(answer)
        if len(sentences) == 0:
            continue

        for sentence in sentences:

            sentence_text = ' '.join(sentence)
            sentence_vector = model.get_sentence_vector(sentence_text)
            answer_vector.append(sentence_vector)
        
        answer_vector = np.array(answer_vector)
        answer_vector = np.mean(answer_vector, axis=0)

        answers[document_id, :] = answer_vector
    
    np.save(path, answers)


def create_answer_matrix(model, df, path):
    
    max_sentence = 0
    for _, answer in enumerate(df["answer"]):
        max_sentence = max(len(process_answer(answer)), max_sentence)
    
    print(f"Max sentences: {max_sentence}")
    
    answers = np.zeros((len(df['answer']), max_sentence * 300))

    for document_id, answer in enumerate(df["answer"]):
        
        answer_matrix = np.zeros((max_sentence, 300))

        sentences = process_answer(answer)
        if len(sentences) == 0:
            continue

        for index, sentence in enumerate(sentences):
            sentence_text = ''.join(sentence)
            sentence_vector = model.get_sentence_vector(sentence_text)
            answer_matrix[index, :] = sentence_vector
        
        answers[document_id, :] = answer_matrix.flatten()
    
    np.save(path, answers)
                

def time_nearest_word(model, word):
    start = time.perf_counter()
    for _ in range(100):
        temp = model.get_nearest_neighbors(word, k=1)[0][1]
    end = time.perf_counter()

    print(f"nearest: {(end - start) / 100}")

def construct_document_matrix(df, model):
    n_documents = df["answer"].size
    byte_terms = dict()
    term_id = 0

    print("Scanning terms")

    for document_id, answer in enumerate(df["answer"]):    
        for sentence in process_answer(answer):
            for term in sentence:
                vector_term = model[term]
                byte_term = vector_term.tobytes()

                if byte_term not in byte_terms:
                    byte_terms[byte_term] = term_id
                    term_id += 1


    print("Creating matrix")
    # Create matrix 
    M = np.zeros((n_documents, term_id))
    for document_id, answer in enumerate(df["answer"]):
    
        for sentence in process_answer(answer):
            for term in sentence:
                vector_term = model[term]
                byte_term = vector_term.tobytes()
                term_id = byte_terms[byte_term]
                
                M[document_id, term_id] = M[document_id, term_id] + 1 
    
    return M


#create_corpus(load_data("q_2_k_24.json"), "q_2_24.txt")

retrain("cc.sk.300.bin", "cc.all.bin", "all.txt")

# M = construct_document_matrix(load_data("q_2_k_24.json"), load_model("wiki"));

# model = load_gensim("cc.sk.300.bin")
# print("processing")
# create_answer_vectors(
#     model,
#     pd.read_json(join(CWD, 'data', 'q_2_k_12.json')),
#     join(CWD, 'data', '12_cc_vector.npy')
# )

# create_answer_matrix(
#     model,
#     pd.read_json(join(CWD, 'data', 'q_2_k_12.json')),
#     join(CWD, 'data', '12_cc_matrix.npy')
# )

# create_answer_vectors(
#     model,
#     load_data("q_2_k_24.json"),
#     join(CWD, 'data', 'test_original_vector.npy')
# )

# create_answer_matrix(
#     model,
#     load_data("q_2_k_24.json"),
#     join(CWD, 'data', 'test_original_matrix.npy')
# )


# process_nearest_terms(load_data("q_2_k_12.json"), load_model("cc.sk.300.bin"), join(CWD, 'nearest', "q_2_k_12_cc.csv"))
# process_nearest_terms(load_data("q_2_k_12.json"), load_model("wiki.sk.bin"), join(CWD, 'nearest', "q_2_k_12_wiki.csv"))
