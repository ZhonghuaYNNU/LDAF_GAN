from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

def lnc_seq():
    common_texts = []
    every_lnc_seq = []
    f = open("xulie_test")
    lnc_seq = f.readlines()
    for seq_i in range(len(lnc_seq)):
        seq = list(lnc_seq[seq_i])
        for j in range(0, len(seq), 1):
            if j <= len(seq)-3:
                group_3 = seq[j]+seq[j+1]+seq[j+2]
                group_3 = str(group_3)
                every_lnc_seq.append(group_3)
        common_texts.append(list(every_lnc_seq))
        every_lnc_seq = []

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]

    model = Doc2Vec(documents, vector_size=1, window=2, min_count=1, workers=4, epochs=50)  # 设置模型参数
    model.build_vocab(documents, update=True)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    f1 = open("xulie")
    lnc_seq_pre = f1.readlines()
    max_len = 0
    final = []
    row = 0
    for i in range(len(lnc_seq_pre)):
        seq_mine = list(lnc_seq_pre[i].replace('\n', ''))
        length = len(seq_mine)
        if length > max_len:
            max_len = length
    max_len = max_len-3+1
    zero_matrix = np.zeros((max_len, 1))

    for seq_pre in range(len(lnc_seq_pre)):
        seq_mine = list(lnc_seq_pre[seq_pre].replace('\n', ''))
        zero_matrix[:, :] = 0
        for jj in range(0, len(seq_mine), 1):
            if jj <= len(seq_mine)-3:
                group_3_mine = seq_mine[jj]+seq_mine[jj+1]+seq_mine[jj+2]
                group_3_mine = str(group_3_mine)
                group_3_vector = model[group_3_mine]
                zero_matrix[row, :] = group_3_vector
            row = row + 1
        final.append(zero_matrix)
        row = 0

    return final

if __name__ == '__main__':
    final = lnc_seq()
    print(final)
