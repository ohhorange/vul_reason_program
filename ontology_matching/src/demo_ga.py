import numpy as np
import Levenshtein as lev
from nltk.corpus import wordnet as wn
from nltk.tokenize import wordpunct_tokenize
from ga import genetic_algorithm
from jaro_winkler.jaro_winkler import JaroWinkler

JA=JaroWinkler()

from gensim.models import KeyedVectors
# 加载预训练的Word2Vec模型
model_path = '../GoogleNews-vectors-negative300.bin'
Word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=None)

def needlemanwunsch_distance(string1, string2, gap):
    if string1 == None or string2 == None:
        return 1.0
    n = len(string1)
    m = len(string2)
    if n == 0 or m == 0:
        return 1.0
    
    p = []
    d = []
    
    for i in range(n+1):
        p.append(i * gap)
    for j in range(1, m+1):
        t_j = string2[j-1]
        d.append(j * gap)
        
        for i in range(1, n+1):
            if string1[i-1] == t_j:
                cost = 0
            else:
                cost = 1
            d.append(min(min(d[i-1] + gap, p[i] + gap), p[i-1] + cost))
        d_temp = p
        p = d
        d = d_temp
        d = []
    min_value = min(n, m)
    diff = max(n, m) - min_value
    return float(p[n]) / float(min_value + diff * gap)

def levenshtein_distance(string1, string2):
    return needlemanwunsch_distance(string1, string2, 1)


def substring_distance(string1, string2):
    if string1 == None or string2 == None:
        return 1.0
    
    len1 = len(string1)
    len2 = len(string2)
    
    if len1 == 0 and len2 == 0:
        return 0.0
    if len1 == 0 or len2 == 0:
        return 1.0
    best = 0
    i = 0
    while i < len1 and len1 - 1 > best:
        j = 0
        while len2 - j > best:
            k = i;   
            while j < len2 and string1[k] != string2[j]:
                j += 1
            if j != len2:
                j += 1
                k += 1
                while j < len2 and k < len1 and string1[k] == string2[j]:  
                    j += 1 
                    k += 1
                best = max(best, k-i)
        i += 1
    return 1.0 - (float(2 * best) / float(len1 + len2))
    

def basic_synonym_distance(string1, string2):
        string1 = string1.lower()
        string2 = string2.lower()
        
        dist_subs = substring_distance(string1, string2)
        synsets = wn.synsets(string1, wn.NOUN)
        if len(synsets) == 0:
            tokens = wordpunct_tokenize(string1)
            for token in tokens:
                synsets = wn.synsets(string1, wn.NOUN)
                if len(synsets) > 0:
                    break
        if len(synsets) > 0:
            for synset in synsets:
                for lemma in synset.lemmas():
                    dist = substring_distance(lemma.name(), string2)
                    if (dist < dist_subs):
                        dist_subs = dist
        return dist_subs

def Word2vec_similarity(string1, string2):
    if string1 in Word2vec.vocab and string2 in Word2vec.vocab:
        return Word2vec.similarity(string1, string2)
    else:
        return 0.0


def label_sim_matrix(src_labels, target_labels):
    '''
    :param src_labels: labels of source ontology
    :param target_labels: labels of target ontology
    :return: the matrix of similarity
    '''
    s_len = len(src_labels)
    t_len = len(target_labels)
    mat = np.zeros([s_len, t_len])
    with open("./output.txt","w") as f:
        for i in range(s_len):
            for j in range(t_len):
                sim1=1-levenshtein_distance(src_labels[i].lower(), target_labels[j].lower())
                sim2=1-basic_synonym_distance(src_labels[i].lower(),target_labels[j].lower())
                sim3=Word2vec_similarity(src_labels[i].lower(),target_labels[j].lower())
                w1,w2,w3=genetic_algorithm(sim1,sim2,sim3)
                mat[i][j] = sim1*1/3+sim2*1/3+sim3*1/3

                f.write("labels1:{} labels2:{}\nlevenshtein:{} synonym_distance:{} Jaro_winkler:{} res_similarity:{}\n".format(src_labels[i].lower(),target_labels[j].lower(),sim1,sim2,sim3,mat[i][j]))
            # mat[i][j]=max(sim1,sim2,sim3)
    return mat

def matching_by_similarity_threshold(src_rdf, target_rdf, threshold):
    print('Similarity threshold: {}'.format(threshold))
    # (entity1, entity2, measure, relation): entity1 is from 101
    matching_pairs = []

    # classes
    sim_mat = label_sim_matrix(src_rdf.class_labels, target_rdf.class_labels)
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if sim_mat[i][j] >= threshold:
                print(src_rdf.class_uris[i], target_rdf.class_uris[j])
                matching_pairs.append((src_rdf.class_uris[i], target_rdf.class_uris[j], sim_mat[i][j], '='))

    # attributes
    sim_mat = label_sim_matrix(src_rdf.attribute_labels, target_rdf.attribute_labels)
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if sim_mat[i][j] >= threshold:
                matching_pairs.append((src_rdf.attribute_uris[i], target_rdf.attribute_uris[j], sim_mat[i][j], '='))

    # relationships
    sim_mat = label_sim_matrix(src_rdf.relation_labels, target_rdf.relation_labels)
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if sim_mat[i][j] >= threshold:
                matching_pairs.append((src_rdf.relation_uris[i], target_rdf.relation_uris[j],sim_mat[i][j], '='))

    return matching_pairs
