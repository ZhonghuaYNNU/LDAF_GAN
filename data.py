from collections import defaultdict
import torch
import pandas as pd

def loadlnc_seq(trainFile, splitMark):
    print(trainFile)

    lnc_seq = pd.DataFrame(columns=["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9",
                                    "L10", "L11", "L12", "L13", "L14", "L15", "L16", "L17", "L18", "L19",
                                    "L20", "L21", "L22", "L23", "L24", "L25", "L26", "L27", "L28", "L29",
                                    "L30", "L31", "L32", "L33", "L34", "L35", "L36", "L37", "L38", "L39",
                                    "L40", "L41", "L42", "L43", "L44", "L45", "L46", "L47", "L48", "L49",
                                    "L50", "L51", "L52", "L53", "L54", "L55", "L56", "L57", "L58", "L59",
                                    "L60", "L61", "L62", "L63", "L64"])
    index = 0

    for line in open(trainFile):
        L1, L2, L3, L4, L5, L6, L7, L8, L9,\
        L10, L11, L12, L13, L14, L15, L16, L17, L18, L19,\
        L20, L21, L22, L23, L24, L25, L26, L27, L28, L29,\
        L30, L31, L32, L33, L34, L35, L36, L37, L38, L39,\
        L40, L41, L42, L43, L44, L45, L46, L47, L48, L49,\
        L50, L51, L52, L53, L54, L55, L56, L57, L58, L59,\
        L60, L61, L62, L63, L64 = line.strip().split(splitMark)
       
        lnc_seq.loc['%d' % index] = [L1, L2, L3, L4, L5, L6, L7, L8, L9,\
        L10, L11, L12, L13, L14, L15, L16, L17, L18, L19,\
        L20, L21, L22, L23, L24, L25, L26, L27, L28, L29,\
        L30, L31, L32, L33, L34, L35, L36, L37, L38, L39,\
        L40, L41, L42, L43, L44, L45, L46, L47, L48, L49,\
        L50, L51, L52, L53, L54, L55, L56, L57, L58, L59,\
        L60, L61, L62, L63, L64]

        index = index + 1

        lnc_seq.to_csv("lncRNA_embedding_save.csv", index=False)

    return lnc_seq


def loadTrainingData(trainFile,splitMark):
    trainSet = defaultdict(list)
    max_lnc_id = -1
    max_dis_id = -1
    for line in open(trainFile):
        lnc_Id, dis_Id = line.strip().split(splitMark)
        lnc_Id = int(lnc_Id)
        dis_Id = int(dis_Id)
        trainSet[lnc_Id].append(dis_Id)
        max_lnc_id = max(lnc_Id, max_lnc_id)
        max_dis_id = max(dis_Id, max_dis_id)

    lncCount = max_lnc_id + 1
    disCount = max_dis_id + 1

    return trainSet, lncCount, disCount

def loadTestData(testFile,splitMark):
    testSet = defaultdict(list)
    max_lnc_id = -1
    max_dis_id = -1
    for line in open(testFile):
        lnc_Id, dis_Id = line.strip().split(splitMark)
        lnc_Id = int(lnc_Id)
        dis_Id = int(dis_Id)
        testSet[lnc_Id].append(dis_Id)
        max_lnc_id = max(lnc_Id, max_lnc_id)
        max_dis_id = max(dis_Id, max_dis_id)
    lncCount = max_lnc_id + 1
    disCount = max_dis_id + 1

    return testSet, lncCount, disCount


def to_Vectors(trainSet, lncCount, disCount, lncList_test, mode):
    
    testMaskDict = defaultdict(lambda: [0] * disCount)
    batchCount = lncCount
    trainDict = defaultdict(lambda: [0] * disCount)
    for lncId, i_list in trainSet.items():
        for disId in i_list:
            testMaskDict[lncId][disId] = -99999
            if mode == "lncBased":
                trainDict[lncId][disId] = 1.0
            else:
                trainDict[disId][lncId] = 1.0

    trainVector = []

    for batchId in range(batchCount):
        trainVector.append(trainDict[batchId])

    testMaskVector = []
    for lncId in lncList_test:
        testMaskVector.append(testMaskDict[lncId])

    return (torch.Tensor(trainVector)), torch.Tensor(testMaskVector), batchCount

