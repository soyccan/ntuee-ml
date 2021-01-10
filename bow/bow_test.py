import os
import joblib
import torch
from sklearn.feature_extraction.text import CountVectorizer


from rnn.util import *

def bag_of_words(corpus):
    vectorizer = joblib.load('bow.pkl')
    word_vectors = vectorizer.transform(corpus)
    return word_vectors  # sparse matrix


def main():
    # set GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    print("loading data ...")
    X = load_testing_data('../work/a.txt')
    X = bag_of_words([' '.join(ln) for ln in X])

    model = torch.load('../work/model/word2vec-0.82560.model')
    model = model.to(device)
    model.eval()

    ret_output = []
    with torch.no_grad():
        inputs = torch.from_numpy(X.todense()).to(device, dtype=torch.long)
        outputs = model(inputs)
        print(outputs)
        outputs = outputs.squeeze()
        outputs[outputs >= 0.5] = 1  # 大於等於 0.5 為負面
        outputs[outputs < 0.5] = 0  # 小於 0.5 為正面
        ret_output += outputs.int().tolist()

    return ret_output

main()
