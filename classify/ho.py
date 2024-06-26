#!/usr/bin/env python
# coding: utf-8
"""
Hold out classification analysis of BCI Comp IV-2a and Korea datasets
@author: Ravikiran Mane
"""
import numpy as np
import torch
import sys
import os
import time
import xlwt
import csv
import random
import math
import copy

masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo'))
from eegDataset import eegDataset
from baseModel import baseModel
import networks
import transforms
from saveData import fetchData

# reporting settings
debug = False

def ho(datasetId = None, network = None, nGPU = None, subTorun=None):

    #%% Set the defaults use these to quickly run the network
    datasetId = datasetId or 0
    network = network or 'DMSANet'
    nGPU = nGPU or 0
    subTorun= subTorun or None
    selectiveSubs = False
    
    # decide which data to operate on:
    # datasetId ->  0:BCI-IV-2a data,    1: HGD data 2:OpenBMI data
    datasets = ['bci42a', 'hgd','korea']
    
    #%% Define all the model and training related options here.
    config = {}

    # Data load options:
    config['preloadData'] = False # whether to load the complete data in the memory

    # Random seed
    config['randSeed']  = 20190821
    
    # Network related details
    config['network'] = network
    config['batchSize'] = 16
    
    if datasetId == 1:
        config['modelArguments'] = {'nChan': 44, 'nTime': 1000, 'dropoutP': 0.5,
                                    'nBands': 9, 'm': 32, 'temporalLayer': 'LogVarLayer',
                                    'nClass': 4, 'doWeightNorm': True}
    elif datasetId == 0:
        config['modelArguments'] = {'nChan': 22, 'nTime': 1000, 'dropoutP': 0.5,
                                    'nBands':9, 'm' : 32, 'temporalLayer': 'LogVarLayer',
                                    'nClass': 4, 'doWeightNorm': True}
    elif datasetId == 2:
        config['modelArguments'] = {'nChan': 20, 'nTime': 1000, 'dropoutP': 0.5,
                                    'nBands': 9, 'm': 32, 'temporalLayer': 'LogVarLayer',
                                    'nClass': 2, 'doWeightNorm': True}
    
    # Training related details    
    config['modelTrainArguments'] = {'stopCondi':  {'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': 1500, 'varName' : 'epoch'}},
                                                       'c2': {'NoDecrease': {'numEpochs' : 200, 'varName': 'valInacc'}} } }},
          'classes': [0,1,2,3], 'sampler' : 'RandomSampler', 'loadBestModel': True,
          'bestVarToCheck': 'valInacc', 'continueAfterEarlystop':True,'lr': 1e-3}
    if datasetId ==2:
        config['modelTrainArguments']['classes'] = [0,1] # 2 class data

    config['transformArguments'] = None

    # add some more run specific details.
    config['cv'] = 'trainTest'
    config['kFold'] = 1
    config['data'] = 'raw'
    config['subTorun'] = subTorun
    config['trainDataToUse'] = 1.0   # How much data to use for training
    config['validationSet'] = 0.2   # how much of the training data will be used a validation set

    # network initialization details:
    config['loadNetInitState'] = True
    config['pathNetInitState'] = config['network'] + '_'+ str(datasetId)

    #%% Define data path things here. Do it once and forget it!
    # Input data base folder:
    toolboxPath = os.path.dirname(masterPath)
    config['inDataPath'] = os.path.join(toolboxPath, 'data')



    # # Input data datasetId folders
    # modeInFol = 'rawPython'
    # Input data datasetId folders
    if 'FBCNet' in config['network']:
        modeInFol = 'multiviewPython'  # FBCNet uses multi-view data
    else:
        modeInFol = 'rawPython'

    # set final input location
    config['inDataPath'] = os.path.join(config['inDataPath'], datasets[datasetId], modeInFol)

    # Path to the input data labels file
    config['inLabelPath'] = os.path.join(config['inDataPath'], 'dataLabels.csv')

    # Output folder:
    # Lets store all the outputs of the given run in folder.
    config['outPath'] = os.path.join(toolboxPath, 'output')
    config['outPath'] = os.path.join(config['outPath'], datasets[datasetId], 'ses2Test')

    # Network initialization:
    config['pathNetInitState'] = os.path.join(masterPath, 'netInitModels', config['pathNetInitState']+'.pth')
    # check if the file exists else raise a flag
    config['netInitStateExists'] = os.path.isfile(config['pathNetInitState'])
    
    #%% Some functions that should be defined here

    def setRandom(seed):
        '''
        Set all the random initializations with a given seed

        '''
        # Set np
        np.random.seed(seed)

        # Set torch
        torch.manual_seed(seed)

        # Set cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def excelAddData(worksheet, startCell, data, isNpData = False):
        '''
            Write the given max 2D data to a given given worksheet starting from the start-cell.
            List will be treated as a row.
            List of list will be treated in a matrix format with inner list constituting a row.
            will return the modified worksheet which needs to be written to a file
            isNpData flag indicate whether the incoming data in the list is of np data-type
        '''
        #  Check the input type.
        if type(data) is not list:
            data = [[data]]
        elif type(data[0]) is not list:
            data = [data]
        else:
            data = data

        # write the data. starting from the given start cell.
        rowStart = startCell[0]
        colStart = startCell[1]

        for i, row in enumerate(data):
            for j, col in enumerate(row):
                if isNpData:
                    worksheet.write(rowStart+i, colStart+j, col.item())
                else:
                    worksheet.write(rowStart+i, colStart+j, col)

        return worksheet

    def dictToCsv(filePath, dictToWrite):
    	"""
    	Write a dictionary to a given csv file
    	"""
    	with open(filePath, 'w') as csv_file:
    		writer = csv.writer(csv_file)
    		for key, value in dictToWrite.items():
    			writer.writerow([key, value])

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    #%% create output folder
    # based on current date and time -> always unique!
    randomFolder = str(time.strftime("%Y-%m-%d--%H-%M", time.localtime()))+ '-'+str(random.randint(1,1000))

    config['outPath'] = os.path.join(config['outPath'],config['network'],str(config['trainDataToUse']), randomFolder,'')
    # create the path
    if not os.path.exists(config['outPath']):
        os.makedirs(config['outPath'])
    print('Outputs will be saved in folder : ' + config['outPath'])

    # Write the config dictionary
    dictToCsv(os.path.join(config['outPath'],'config.csv'), config)

    #%% Check and compose transforms
    if config['transformArguments'] is not None:
        if len(config['transformArguments']) >1 :
            transform = transforms.Compose([transforms.__dict__[key](**value) for key, value in config['transformArguments'].items()])
        else:
            transform = transforms.__dict__[list(config['transformArguments'].keys())[0]](**config['transformArguments'][list(config['transformArguments'].keys())[0]])
    else:
        transform = None

    #%% check and Load the data
    print('Data loading in progress')
    fetchData(os.path.dirname(config['inDataPath']), datasetId) # Make sure that all the required data is present!
    data = eegDataset(dataPath = config['inDataPath'], dataLabelsPath= config['inLabelPath'], preloadData = config['preloadData'], transform= transform)
    print('Data loading finished')

    #%% Check and load the model
    #import networks
    if config['network'] in networks.__dict__.keys():
        network = networks.__dict__[config['network']]
    else:
        raise AssertionError('No network named '+ config['network'] + ' is not defined in the networks.py file')

    # Load the net and print trainable parameters:
    net = network(**config['modelArguments'])
    print('Trainable Parameters in the network are: ' + str(count_parameters(net)))

    #%% check and load/save the the network initialization.
    if config['loadNetInitState']:
        if config['netInitStateExists']:
            netInitState = torch.load(config['pathNetInitState'])
        else:
            setRandom(config['randSeed'])
            net = network(**config['modelArguments'])
            netInitState = net.to('cpu').state_dict()
            torch.save(netInitState, config['pathNetInitState'])

   #%% Find all the subjects to run 
    subs = sorted(set([d[3] for d in data.labels]))

    nSub = len(subs)

    ## Set sub2run
    if selectiveSubs:
        config['subTorun'] = config['subTorun']
    else:
        if config['subTorun']:
            config['subTorun'] = list(range(config['subTorun'][0], config['subTorun'][1]))
        else:
            config['subTorun'] = list(range(nSub))

    #%% Let the training begin
    trainResults = []
    valResults = []
    testResults = []
    trainTime = []

    for iSub, sub in enumerate(subs):
        
        if iSub not in config['subTorun']:
            continue
        
        start = time.time()
        
        # extract subject data
        subIdx = [i for i, x in enumerate(data.labels) if x[3] in sub]
        subData = copy.deepcopy(data)
        subData.createPartialDataset(subIdx, loadNonLoadedData = True)
        
        trainData = copy.deepcopy(subData)
        testData = copy.deepcopy(subData)
        
        # Isolate the train -> session 0 and test data-> session 1
        if len(subData.labels[0])>4:
            idxTrain = [i for i, x in enumerate(subData.labels) if x[4] == '0' ]
            idxTest = [i for i, x in enumerate(subData.labels) if x[4] == '1' ]
        else:
            raise ValueError("The data can not be divided based on the sessions")
        
        testData.createPartialDataset(idxTest)
        trainData.createPartialDataset(idxTrain)
        
        # extract the desired amount of train data: 
        trainData.createPartialDataset(list(range(0, math.ceil(len(trainData)*config['trainDataToUse']))))

        # isolate the train and validation set
        valData = copy.deepcopy(trainData)
        valData.createPartialDataset(list( range( 
            math.ceil(len(trainData)*(1-config['validationSet'])) , len(trainData))))
        trainData.createPartialDataset(list(range(0, math.ceil(len(trainData)*(1-config['validationSet'])))))
        
        # Call the network for training
        setRandom(config['randSeed'])


        net = network(**config['modelArguments'])
        net.load_state_dict(netInitState, strict=False)
        
        outPathSub = os.path.join(config['outPath'], 'sub'+ str(iSub))

        start = time.time()
        model = baseModel(net=net, resultsSavePath=outPathSub, batchSize= config['batchSize'], nGPU = nGPU)
        model.train(trainData, valData, testData, **config['modelTrainArguments'])

        # Time taken
        trainTime.append(time.time() - start)
        print("Time taken = " + str(time.time() - start))
        
        # extract the important results.
        trainResults.append([d['results']['trainBest'] for d in model.expDetails])
        valResults.append([d['results']['valBest'] for d in model.expDetails])
        testResults.append([d['results']['test'] for d in model.expDetails])
        
        # save the results
        results = {'train:' : trainResults[-1], 'val: ': valResults[-1], 'test': testResults[-1]} 
        dictToCsv(os.path.join(outPathSub,'results.csv'), results)
        
        # Time taken
        print("Time taken = "+ str(time.time()-start))
 
    #%% Extract and write the results to excel file.

    # lets group the results for all the subjects using experiment.
    # the train, test and val accuracy and cm will be written

    trainAcc = [[r['acc'] for r in result] for result in trainResults]
    trainAcc = list(map(list, zip(*trainAcc)))
    valAcc = [[r['acc'] for r in result] for result in valResults]
    valAcc = list(map(list, zip(*valAcc)))
    testAcc = [[r['acc'] for r in result] for result in testResults]
    testAcc = list(map(list, zip(*testAcc)))

    print("Results sequence is train, val , test")
    print(trainAcc)
    print(valAcc)
    print(testAcc)

    # kappa
    trainKappa = [[r['kappa'] for r in result] for result in trainResults]
    trainKappa = list(map(list, zip(*trainKappa)))
    valKappa = [[r['kappa'] for r in result] for result in valResults]
    valKappa = list(map(list, zip(*valKappa)))
    testKappa = [[r['kappa'] for r in result] for result in testResults]
    testKappa = list(map(list, zip(*testKappa)))

    print(trainKappa)
    print(valKappa)
    print(testKappa)

    # f1
    trainF1 = [[r['f1'] for r in result] for result in trainResults]
    trainF1 = list(map(list, zip(*trainF1)))
    valF1 = [[r['f1'] for r in result] for result in valResults]
    valF1 = list(map(list, zip(*valF1)))
    testF1 = [[r['f1'] for r in result] for result in testResults]
    testF1 = list(map(list, zip(*testF1)))

    print(trainF1)
    print(valF1)
    print(testF1)

    # recall
    trainRecall = [[r['recall'] for r in result] for result in trainResults]
    trainRecall = list(map(list, zip(*trainRecall)))
    valRecall = [[r['recall'] for r in result] for result in valResults]
    valRecall = list(map(list, zip(*valRecall)))
    testRecall = [[r['recall'] for r in result] for result in testResults]
    testRecall = list(map(list, zip(*testRecall)))

    # precision
    trainPre = [[r['pre'] for r in result] for result in trainResults]
    trainPre = list(map(list, zip(*trainPre)))
    valPre = [[r['pre'] for r in result] for result in valResults]
    valPre = list(map(list, zip(*valPre)))
    testPre = [[r['pre'] for r in result] for result in testResults]
    testPre = list(map(list, zip(*testPre)))

    print(trainPre)
    print(valPre)
    print(testPre)



    # append the confusion matrix
    trainCm = [[r['cm'] for r in result] for result in trainResults]
    trainCm = list(map(list, zip(*trainCm)))
    trainCm = [np.concatenate(tuple([cm for cm in cms]), axis = 1) for cms in trainCm]

    valCm = [[r['cm'] for r in result] for result in valResults]
    valCm = list(map(list, zip(*valCm)))
    valCm = [np.concatenate(tuple([cm for cm in cms]), axis = 1) for cms in valCm]


    testCm = [[r['cm'] for r in result] for result in testResults]
    result_cm = np.sum(testCm, axis=0)

    testCm = list(map(list, zip(*testCm)))
    testCm = [np.concatenate(tuple([cm for cm in cms]), axis = 1) for cms in testCm]

    print(trainCm)
    print(valCm)
    print(testCm)
    # %% txt writing
    resultName = os.path.join(config['outPath'], 'results.txt')

    with open(resultName, 'w', encoding='utf-8') as file:
        file.write("---------------Acc---------------"+"\n")
        file.write("ACC"+"\n")
        file.write(str(testAcc) + "\n")
        file.write("Avg:" +str(np.mean([sublist[0] for sublist in testAcc]))+ "\n")
        file.write("Std:" + str(np.std([sublist[0] for sublist in testAcc])) + "\n")
        file.write("---------------Kappa---------------" + "\n")
        file.write("Kappa" + "\n")
        file.write(str(testKappa) + "\n")
        file.write("Avg:" + str(np.mean([sublist[0] for sublist in testKappa])) + "\n")
        file.write("Std:" + str(np.std([sublist[0] for sublist in testKappa])) + "\n")
        file.write("---------------F1---------------" + "\n")
        file.write("F1" + "\n")
        file.write(str(testF1) + "\n")
        file.write("Avg:" + str(np.mean([sublist[0] for sublist in testF1])) + "\n")
        file.write("Std:" + str(np.std([sublist[0] for sublist in testF1])) + "\n")
        file.write("---------------Pre---------------" + "\n")
        file.write("Pre" + "\n")
        file.write(str(testPre) + "\n")
        file.write("Avg:" + str(np.mean([sublist[0] for sublist in testPre])) + "\n")
        file.write("Std:" + str(np.std([sublist[0] for sublist in testPre])) + "\n")
        file.write("---------------CM---------------" + "\n")
        file.write("CM" + "\n")
        file.write(str(testCm) + "\n")

        file.write("---------------TrainTime---------------" + "\n")
        file.write("TrainTime" + "\n")
        file.write(str(trainTime) + "\n")
        file.write("Avg:" + str(np.mean(trainTime)) + "\n\n")

        file.write('Trainable Parameters in the network are: ' + str(count_parameters(net)))






    #%% Excel writing
    book = xlwt.Workbook(encoding="utf-8")
    for i, res in enumerate(trainAcc):
        sheet1 = book.add_sheet('exp-'+str(i+1), cell_overwrite_ok=True)
        sheet1 = excelAddData(sheet1, [0,0], ['SubId', 'trainAcc', 'valAcc', 'testAcc',
                                              'trainKappa','valKappa','testKappa',
                                              'trainF1','valF1','testF1',
                                              'trainRecall','valRecall','testRecall',
                                              'trainPre','valPre','testPre'])
        sheet1 = excelAddData(sheet1, [1,0], [[sub] for sub in subs])
        sheet1 = excelAddData(sheet1, [1,1], [[acc] for acc in trainAcc[i]], isNpData= True)
        sheet1 = excelAddData(sheet1, [1,2], [[acc] for acc in valAcc[i]], isNpData= True)
        sheet1 = excelAddData(sheet1, [1,3], [[acc] for acc in testAcc[i]], isNpData= True)

        # kappa
        sheet1 = excelAddData(sheet1, [1, 4], [[kappa] for kappa in trainKappa[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 5], [[kappa] for kappa in valKappa[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 6], [[kappa] for kappa in testKappa[i]], isNpData=True)
        # f1
        sheet1 = excelAddData(sheet1, [1, 7], [[f1] for f1 in trainF1[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 8], [[f1] for f1 in valF1[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 9], [[f1] for f1 in testF1[i]], isNpData=True)
        # recall
        sheet1 = excelAddData(sheet1, [1, 10], [[recall] for recall in trainRecall[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 11], [[recall] for recall in valRecall[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 12], [[recall] for recall in testRecall[i]], isNpData=True)
        # precision
        sheet1 = excelAddData(sheet1, [1, 13], [[pre] for pre in trainPre[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 14], [[pre] for pre in valPre[i]], isNpData=True)
        sheet1 = excelAddData(sheet1, [1, 15], [[pre] for pre in testPre[i]], isNpData=True)

        # time
        sheet1 = excelAddData(sheet1, [1, 16], [[time] for time in trainTime])

        # write the cm
        for isub, sub in enumerate(subs):
            sheet1 = excelAddData(sheet1, [len(trainAcc[0])+5,0+isub*len( config['modelTrainArguments']['classes'])], sub)
        sheet1 = excelAddData(sheet1, [len(trainAcc[0])+6,0], ['train CM:'])
        sheet1 = excelAddData(sheet1, [len(trainAcc[0])+7,0], trainCm[i].tolist(), isNpData= False)
        sheet1 = excelAddData(sheet1, [len(trainAcc[0])+11,0], ['val CM:'])
        sheet1 = excelAddData(sheet1, [len(trainAcc[0])+12,0], valCm[i].tolist(), isNpData= False)
        sheet1 = excelAddData(sheet1, [len(trainAcc[0])+17,0], ['test CM:'])
        sheet1 = excelAddData(sheet1, [len(trainAcc[0])+18,0], testCm[i].tolist(), isNpData= False)

        sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 22, 0], ['total CM:'])
        sheet1 = excelAddData(sheet1, [len(trainAcc[0]) + 23, 0], result_cm[i].tolist(), isNpData=False)

    book.save(os.path.join(config['outPath'], 'results.xls'))

if __name__ == '__main__':
    arguments = sys.argv[1:]
    count = len(arguments)

    if count >0:
        datasetId = int(arguments[0])
    else:
        datasetId = None

    if count > 1:
        network = str(arguments[1])
    else:
        network = None

    if count >2:
        nGPU = int(arguments[2])
    else:
        nGPU = None

    if count >3:
        subTorun = [int(s) for s in str(arguments[3]).split(',')]

    else:
        subTorun = None
    ho(datasetId=0, network='DMSANet', nGPU='0', subTorun=subTorun)
