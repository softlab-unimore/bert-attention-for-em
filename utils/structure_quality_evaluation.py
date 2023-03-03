import os
import pandas as pd
import json
import numpy as np
import csv
from collections import Counter
import math
from statistics import mean
import itertools
from scipy.stats import entropy

pathResultOut = '..\\results\\models\\'
pathTest = '..\\data\\'

def createJSON(nameDataset, type):
    pathData = nameDataset.split('_')
    pathFileOutput = pathResultOut + nameDataset + '_sent_pair_tuned\\' + 'groupFind' + type + '.txt'
    pathFileTest = pathTest + pathData[0] + '\\' + pathData[1] + '\\' + 'test.csv'
    pathFileJson = pathTest + pathData[0] + '\\' + pathData[1] + '\\' + 'outputJson' + type + '.json'
    testFile = pd.read_csv(pathFileTest)
    print(testFile)
    dictGroup = {}
    with open(pathFileOutput, encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if idx % 3 == 0:
                label = testFile.iloc[[int(line) - 1]]['label']
                indice = int(line) - 1
                dictGroup[indice] = []
                idxPlus = 0
            else:
                listAttr = []
                idxPlus += 1
                if idxPlus == 2:
                    for lineIns in line.split('],'):
                        listApp = []
                        for attr in lineIns.split(','):
                            word = attr.replace('[', '')
                            word = word.replace(']', '')
                            # word = word.replace('left_','')
                            # word = word.replace('right_','')
                            word = word.replace('\n', '')
                            listApp.append(word)
                        listAttr.append(listApp)
                    dictGroup[indice] = {'listAttr': listAttr, 'label': int(label)}
        with open(pathFileJson, "w") as outfile:
            json.dump(dictGroup, outfile)


def countEntropy(listAttrCompl):
    listSingle = [x.split('_')[1] for x in listAttrCompl]
    counts = Counter(listSingle)
    # Compute the probability distribution
    p = [count / len(listSingle) for count in counts.values()]
    # Compute the entropy
    entr = entropy(p)

    return entr


def countOccurrence(listAttrCompl):
    for index in range(len(listAttrCompl)):
        attr = listAttrCompl[index]
        if attr.split('_')[0] == 'left':
            word = 'right_'
        elif attr.split('_')[0] == 'right':
            word = 'left_'
        else:
            continue
        if word + attr.split('_')[1] in listAttrCompl:
            listApp = []
            for xW in listAttrCompl:
                if len(xW.split('_')) == 2:
                    if xW.split('_')[1] != attr.split('_')[1]:
                        listApp.append(xW)
                    else:
                        listApp.append(attr.split('_')[1])
                else:
                    listApp.append(xW)
            listAttrCompl = listApp
    listAttrApp = []
    for x in listAttrCompl:
        if x.split('_')[0] != 'left' and x.split('_')[0] != 'right':
            if listAttrApp == '':
                print(' dsadas')
            listAttrApp.append(x)

    if len(listAttrApp) == 1:
        print('adasdsad')
    return len(listAttrApp) / len(listAttrCompl), listAttrApp


def getDictAttr(listAttr):
    dictVal = {}
    for nameAttr in listAttr:
        if 'left' in nameAttr and nameAttr != 'left_id':
            dictVal[nameAttr.split('_')[1]] = 0
    return dictVal


def measureQuality(listAttr):
    entropyInt = 0
    sumOcc = 0
    listBestAttr = []
    listRipet = set()
    for listAttrCompl in listAttr:
        if listAttrCompl != ['']:
            countElement = Counter(listAttrCompl)
            if len(set(listAttrCompl)) > 1:
                entropyInt += countEntropy(listAttrCompl)
            max_keys = [key for key, value in countElement.items() if value == max(countElement.values())]
            valMedio, listAppComp = countOccurrence(listAttrCompl)
            sumOcc += valMedio
            listRipetIntern = set()
            for al in listAppComp:
                if al == '':
                    print('cacas')
                listRipetIntern.add(al)
                listRipet.add(al)

            if len(listRipetIntern) > 0:
                max_keys = list(iter(listRipetIntern))
            else:
                max_keys = [x.split('_')[1] if len(x.split('_')) == 2 else x for x in max_keys]
            # sumOcc += countElement[max_keys[0]]/sum(countElement.values())
            listBestAttr.append(max_keys)
        else:
            listBestAttr.append([''])

    averageOcc = sumOcc / len(listAttr)
    totalDiff = 0
    partDiff = 0
    for listDif in list(itertools.product(*listBestAttr)):
        if len(set(list(listDif))) == len(list(listDif)) and '' not in set(list(listDif)):
            totalDiff = 1
        if len(set(list(listDif[:3]))) == 3 and '"' not in set(list(listDif)):
            partDiff = 1

    return entropyInt / len(listAttr), totalDiff, partDiff, averageOcc, listRipet



def getQuality(nameDataset, type):
    # open the file in the write mode
    pathData = nameDataset.split('_')
    pathFileJson = pathTest + pathData[0] + '\\' + pathData[1] + '\\outputJson'+type+'.json'
    print(pathFileJson)
    pathResult = pathResultOut + nameDataset + '_sent_pair_tuned\\'
    isExist = os.path.exists(pathResult)
    if not isExist:
        os.makedirs(pathResult)

    pathResultData = pathResult + '\\' + pathData[1]
    isExist = os.path.exists(pathResultData)
    if not isExist:
        os.mkdir(pathResultData)

    pathFileOutputCsv = pathResultData + '\\' + 'resultSingle' + type + '.csv'
    pathFileOutputCsvComplete = pathResultData + '\\' + 'resultComplete' + type + '.csv'

    pathFileCsv = pathTest + pathData[0] + '\\' + pathData[1] + '\\' + 'train.csv'
    df = pd.read_csv(pathFileCsv)
    getDict = getDictAttr(list(df.head()))
    f = open(pathFileOutputCsv, 'w')
    writer = csv.writer(f)
    writer.writerow(['precision', 'label'])
    with open(pathFileJson) as jsonFile:
        jsonObject = json.load(jsonFile)
        listZer = []
        listUn = []
        listZerMatch = []
        listUnMatch = []
        listGroupUnTot = []
        listGroupUnThree = []
        listGroupZerTot = []
        listGroupZerThree = []
        totDif = 0
        totPartDiff = 0
        allAverageOcc = 0
        for x in jsonObject:
            qualityMeasure, diff, partDiff, averageOcc, listRip = measureQuality(jsonObject[str(x)]['listAttr'])
            for nameRep in listRip:
                if nameRep == '':
                    print('ciao')
                getDict[nameRep] += 1
            totDif += diff
            totPartDiff += partDiff
            allAverageOcc += averageOcc
            valuePrecision = qualityMeasure

            label = jsonObject[str(x)]['label']
            if label == 0:
                listZer.append(qualityMeasure)
                listZerMatch.append(averageOcc)
                listGroupZerTot.append(diff)
                listGroupZerThree.append(partDiff)
            else:
                listUn.append(qualityMeasure)
                listUnMatch.append(averageOcc)
                listGroupUnTot.append(diff)
                listGroupUnThree.append(partDiff)
            writer.writerow([str(valuePrecision), str(label)])


        print('Average Entropy 0: ' + str(mean(listZer)))
        print('Average Entropy 1: ' + str(mean(listUn)))
        # print('Average of all groups reveal different attr: ' + str(totDif/(len(listZer) + len(listUn))))
        # print('Average of first 3 groups reveal different attr: ' + str(totPartDiff/(len(listZer) + len(listUn))))
        print('Average of all groups reveal different attr: 0: ' + str(mean(listGroupZerTot)))
        print('Average of all groups reveal different attr 1: ' + str(mean(listGroupUnTot)))
        print('Average of first 3 groups reveal different attr 0: ' + str(mean(listGroupZerThree)))
        print('Average of first 3 groups reveal different attr 1: ' + str(mean(listGroupUnThree)))
        print('Average matching occurrence: ' + str(allAverageOcc / (len(listZer) + len(listUn))))
        print('Average Match 0: ' + str(mean(listZerMatch)))
        print('Average Match 1: ' + str(mean(listUnMatch)))
        print('Value Most Rep: ' + str(getDict))
        print('Value Most Rep: ' + str(len(jsonObject)))

        listApp = []
        for x in getDict.keys():
            listApp.append(getDict[x] / len(jsonObject))

        with open(pathFileOutputCsvComplete, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['AverageEntropy0', 'StdEntropy0', 'AverageEntropy1', 'StdEntropy1', 'AverageDiffAllGroup0',
                 'StdDiffAllGroup0', 'AverageDiffAllGroup1', 'StdDiffAllGroup1',
                 'AverageDiffThreeGroup0', 'StdDiffThreeGroup0', 'AverageDiffThreeGroup1', 'StdDiffThreeGroup1',
                 'AverageMatchOcc', 'AverageMatchOcc0', 'StdMatchOcc0', 'AverageMatchOcc1', 'StdMatchOcc1'] + [x for
                                                                                                               x in
                                                                                                               getDict.keys()] + [
                    'label'])

            writer.writerow([str(mean(listZer)), str(np.std(listZer)), str(mean(listUn)), str(np.std(listUn)),
                             str(mean(listGroupZerTot)), str(np.std(listGroupZerTot)), str(mean(listGroupUnTot)),
                             str(np.std(listGroupUnTot)),
                             str(mean(listGroupZerThree)), str(np.std(listGroupZerThree)),
                             str(mean(listGroupUnThree)), str(np.std(listGroupUnThree)),
                             str(allAverageOcc / (len(listZer) + len(listUn))),
                             str(mean(listZerMatch)), str(np.std(listZerMatch)), str(mean(listUnMatch)),
                             str(np.std(listUnMatch))] +
                            listApp + [label])
    jsonFile.close()
