# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""
Data processing functions for EEG data
@author: Ravikiran Mane
"""
import numpy as np
import mne
from scipy.io import loadmat, savemat
import os
import pickle
import csv
from shutil import copyfile
import sys
import resampy
import shutil
import urllib.request as request
from contextlib import closing
import logging
import sys
import os.path
from collections import OrderedDict
import numpy as np

from braindecode.datasets.bbci import BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
import torch.nn.functional as F
import torch as th
from torch import optim
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.experiment import Experiment
from braindecode.torch_ext.util import np_to_var
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor

from braindecode.datautil.splitters import split_into_two_sets
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import exponential_running_standardize

masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo'))  # To load all the relevant files
from eegDataset import eegDataset
import transforms


def parseBci42aFile(dataPath, labelPath, epochWindow=[0, 4], chans=list(range(22))):
    '''
    Parse the bci42a data file and return an epoched data.

    Parameters
    ----------
    dataPath : str
        path to the gdf file.
    labelPath : str
        path to the labels mat file.
    epochWindow : list, optional
        time segment to extract in seconds. The default is [0,4].
    chans  : list : channels to select from the data.

    Returns
    -------
    data : an EEG structure with following fields:
        x: 3d np array with epoched EEG data : chan x time x trials
        y: 1d np array containing trial labels starting from 0
        s: float, sampling frequency
        c: list of channels - can be list of ints.
    '''
    eventCode = [2]  # start of the trial at t=0
    fs = 250
    offset = 2

    # load the gdf file using MNE
    raw_gdf = mne.io.read_raw_gdf(dataPath, stim_channel="auto")
    raw_gdf.load_data()
    event = mne.events_from_annotations(raw_gdf)
    gdf_events = event[0][:, [0, 2]].tolist()
    event_id = event[1]['768']
    eventCode = [event_id]
    eeg = raw_gdf.get_data()

    # drop channels
    if chans is not None:
        eeg = eeg[chans, :]

    # Epoch the data
    events = [event for event in gdf_events if event[1] in eventCode]
    y = np.array([i[1] for i in events])
    epochInterval = np.array(range(epochWindow[0] * fs, epochWindow[1] * fs)) + offset * fs
    x = np.stack([eeg[:, epochInterval + event[0]] for event in events], axis=2)

    # Multiply the data with 1e6
    x = x * 1e6

    # Load the labels
    y = loadmat(labelPath)["classlabel"].squeeze()
    # change the labels from [1-4] to [0-3]
    y = y - 1

    data = {'x': x, 'y': y, 'c': np.array(raw_gdf.info['ch_names'])[chans].tolist(), 's': fs}
    return data


def parseBci42aDataset(datasetPath, savePath,
                       epochWindow=[0, 4], chans=list(range(22)), verbos=False):
    '''
    Parse the BCI comp. IV-2a data in a MATLAB formate that will be used in the next analysis

    Parameters
    ----------
    datasetPath : str
        Path to the BCI IV2a original dataset in gdf formate.
    savePath : str
        Path on where to save the epoched eeg data in a mat format.
    epochWindow : list, optional
        time segment to extract in seconds. The default is [0,4].
    chans  : list : channels to select from the data.

    Returns
    -------
    None.
    The dataset will be saved at savePath.

    '''
    subjects = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']
    test_subjects = ['A01E', 'A02E', 'A03E', 'A04E', 'A05E', 'A06E', 'A07E', 'A08E', 'A09E']
    subAll = [subjects, test_subjects]
    subL = ['s', 'se']  # s: session 1, se: session 2 (session evaluation)

    print('Extracting the data into mat format: ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    print('Processed data be saved in folder : ' + savePath)

    for iSubs, subs in enumerate(subAll):
        for iSub, sub in enumerate(subs):
            if not os.path.exists(os.path.join(datasetPath, sub + '.mat')):
                raise ValueError('The BCI-IV-2a original dataset doesn\'t exist at path: ' +
                                 os.path.join(datasetPath, sub + '.mat') +
                                 ' Please download and copy the extracted dataset at the above path ' +
                                 ' More details about how to download this data can be found in the Instructions.txt file')

            print('Processing subject No.: ' + subL[iSubs] + str(iSub + 1).zfill(3))
            data = parseBci42aFile(os.path.join(datasetPath, sub + '.gdf'),
                                   os.path.join(datasetPath, sub + '.mat'),
                                   epochWindow=epochWindow, chans=chans)
            savemat(os.path.join(savePath, subL[iSubs] + str(iSub + 1).zfill(3) + '.mat'), data)

def parseHGDFile(dataPath):
    '''

    Args:
        dataPath:
        epochWindow:
        chans:

    Returns:

    '''
    """
        HGD data processing refers to this code: https://github.com/robintibor/high-gamma-dataset/blob/master/example.py
    """
    load_sensor_names = None
    low_cut_hz = 4
    fs = 250.0
    loader = BBCIDataset(dataPath, load_sensor_names=load_sensor_names)

    cnt = loader.load()
    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])

    # Cleaning: First find all trials that have absolute microvolt values
    # larger than +- 800 inside them and remember them for removal later
    clean_ival = [0, 4000]
    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,clean_ival)
    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    # now pick only sensors with C in their name
    # as they cover motor cortex
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']

    cnt = cnt.pick_channels(C_sensors)

    # Resampleing
    cnt = resample_cnt(cnt, fs)
    # Highpassing...
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)

    # Standardizing...
    # cnt = mne_apply(
    #     lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
    #                                               init_block_size=1000,
    #                                               eps=1e-4).T,
    #     cnt)

    # Trial interval
    ival = [0, 4000]
    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    x = dataset.X[clean_trial_mask]
    y = dataset.y[clean_trial_mask]

    data = {'x': x, 'y': y, 'c': C_sensors, 's': fs}
    return data


def parseHGDDataset(datasetPath, savePath):

    subjects = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T', 'A10T', 'A11T', 'A12T', 'A13T',
                'A14T']
    test_subjects = ['A01E', 'A02E', 'A03E', 'A04E', 'A05E', 'A06E', 'A07E', 'A08E', 'A09E', 'A10E', 'A11E', 'A12E',
                     'A13E', 'A14E']
    subAll = [subjects, test_subjects]
    subL = ['s', 'se']  # s: session 1, se: session 2 (session evaluation)

    print('Extracting the data into mat format: ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)  # os.makedirs() 用于创建路径
    print('Processed data be saved in folder : ' + savePath)

    for iSubs, subs in enumerate(subAll):
        for iSub, sub in enumerate(subs):
            # 判断.mat数据是否都在datasetPath中存放
            if not os.path.exists(os.path.join(datasetPath, sub + '.mat')):
                raise ValueError('The HGD original dataset doesn\'t exist at path: ' +
                                 os.path.join(datasetPath, sub + '.mat') +
                                 ' Please download and copy the extracted dataset at the above path ' +
                                 ' More details about how to download this data can be found in the Instructions.txt file')

            print('Processing subject No.: ' + subL[iSubs] + str(iSub + 1).zfill(3))  # 返回指定长度的字符串
            data = parseHGDFile(os.path.join(datasetPath, sub + '.mat'))
            savemat(os.path.join(savePath, subL[iSubs] + str(iSub + 1).zfill(3) + '.mat'), data)



def matToPython(datasetPath, savePath, isFiltered=False):
    '''
    Convert the mat data to eegdataset and save it!

    Parameters
    ----------
    datasetPath : str
        path to mat dataset
    savePath : str
        Path on where to save the epoched eeg data in a eegdataset format.
    isFiltered : bool
        Indicates if the mat data is in the chan*time*trials*FilterBand format.
        default: False

    Returns
    -------
    None.

    '''
    print('Creating python eegdataset with raw data ')
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # load all the mat files
    data = [];
    for root, dirs, files in os.walk(datasetPath):
        files = sorted(files)
        for f in files:
            parD = {}
            parD['fileName'] = f
            parD['data'] = {}
            d = loadmat(os.path.join(root, f),
                        verify_compressed_data_integrity=False)
            # if isFiltered:
            #     parD['data']['eeg'] = np.transpose(d['x'], (2,0,1,3)).astype('float32')
            # else:
            #     parD['data']['eeg'] = np.transpose(d['x'], (2,0,1)).astype('float32')
            # id datasetid=2
            parD['data']['eeg'] = d['x'].astype('float32')

            parD['data']['labels'] = d['y']
            data.append(parD)

    # Start writing the files:
    # save the data in the eegdataloader format.
    # 1 file per sample in a dictionary formate with following fields:
    # id: unique key in 00001 formate
    # data: a 2 dimensional data matrix in chan*time formate
    # label: class of the data
    # Create another separate file to store the epoch info data.
    # This will contain all the intricate data division information.
    # There will be one entry for every data file and will be stored as a 2D array and in csv file.
    # The column names are as follows:
    # id, label -> these should always be present.
    # Optional fields -> subject, session. -> they will be used in data sorting.

    id = 0
    dataLabels = [['id', 'relativeFilePath', 'label', 'subject', 'session']]  # header row

    for i, d in enumerate(data):
        sub = int(d['fileName'][-7:-4])  # subject of the data
        sub = str(sub).zfill(3)

        if d['fileName'][1] == 'e':
            session = 1;
        elif d['fileName'][1] == '-':
            session = int(d['fileName'][2:4])
        else:
            session = 0;

        if len(d['data']['labels']) == 1:
            d['data']['labels'] = np.transpose(d['data']['labels'])

        for j, label in enumerate(d['data']['labels']):
            lab = label[0]
            # get the data
            if isFiltered:
                x = {'id': id, 'data': d['data']['eeg'][j, :, :, :], 'label': lab}
            else:
                x = {'id': id, 'data': d['data']['eeg'][j, :, :], 'label': lab}

            # dump it in the folder
            with open(os.path.join(savePath, str(id).zfill(5) + '.dat'), 'wb') as fp:
                pickle.dump(x, fp)

            # add in data label file
            dataLabels.append([id, str(id).zfill(5) + '.dat', lab, sub, session])

            # increment id
            id += 1
    # Write the dataLabels file as csv
    with open(os.path.join(savePath, "dataLabels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dataLabels)

    # write miscellaneous data info as csv file
    dataInfo = [['fs', 250], ['chanName', 'Check Original File']]
    with open(os.path.join(savePath, "dataInfo.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dataInfo)


def fetchData(dataFolder, datasetId=0):
    '''
    Check if the rawMat, rawPython, and multiviewPython data exists
    if not, then create the above data

    Parameters
    ----------
    dataFolder : str
        The path to the parent dataFolder.
        example: '/home/FBCNetToolbox/data/korea/'
    datasetId : int
        id of the dataset:
            0 : bci42a data (default)
			1 : korea data

    Returns
    -------
    None.

    '''
    print('fetch ssettins: ', dataFolder, datasetId)
    oDataFolder = 'originalData'
    rawMatFolder = 'rawMat'
    rawPythonFolder = 'rawPython'
    multiviewPythonFolder = 'multiviewPython'

    # check that all original data exists
    if not os.path.exists(os.path.join(dataFolder, oDataFolder)):
        if datasetId == 0:
            raise ValueError('The BCI-IV-2a original dataset doesn\'t exist at path: ' +
                             os.path.join(dataFolder, oDataFolder) +
                             ' Please download and copy the extracted dataset at the above path ' +
                             ' More details about how to download this data can be found in the instructions.txt file')
        elif datasetId == 1:
            raise ValueError('The hgd original dataset doesn\'t exist at path: ' +
                             os.path.join(dataFolder, oDataFolder) +
                             ' Please download and copy the extracted dataset at the above path ' +
                             ' More details about how to download this data can be found in the instructions.txt file')
        else :
            raise ValueError('datasetId input error!')
    else:
        oDataLen = len([name for name in os.listdir(os.path.join(dataFolder, oDataFolder))
                        if os.path.isfile(os.path.join(dataFolder, oDataFolder, name))])
        if datasetId == 0 and oDataLen < 36:
            raise ValueError('The BCI-IV-2a dataset at path: ' +
                             os.path.join(dataFolder, oDataFolder) +
                             ' is not complete. Please download and copy the extracted dataset at the above path ' +
                             'The dataset should contain 36 files (18 .mat + 18 .gdf)'
                             ' More details about how to download this data can be found in the instructions.txt file')

        elif datasetId == 1 and oDataLen < 28:
            raise ValueError('The HGD dataset at path: ' +
                             os.path.join(dataFolder, oDataFolder) +
                             ' is not complete. Please download and copy the extracted dataset at the above path ' +
                             'The dataset should contain 28 files (28 .mat)'
                             ' More details about how to download this data can be found in the instructions.txt file')

    # Check if the processed mat data exists:
    if not os.path.exists(os.path.join(dataFolder, rawMatFolder)):
        print('Appears that the raw data exists but its not parsed yet. Starting the data parsing ')
        if datasetId == 0:
            parseBci42aDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder))
        elif datasetId == 1:
            parseHGDDataset(os.path.join(dataFolder, oDataFolder), os.path.join(dataFolder, rawMatFolder))

    # Check if the processed python data exists:
    if not os.path.exists(os.path.join(dataFolder, rawPythonFolder, 'dataLabels.csv')):
        print(
            'Appears that the parsed mat data exists but its not converted to eegdataset yet. Starting this processing')
        matToPython(os.path.join(dataFolder, rawMatFolder), os.path.join(dataFolder, rawPythonFolder))



    print('All the data you need is present! ')
    return 1

