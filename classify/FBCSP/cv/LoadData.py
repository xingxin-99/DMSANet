import mne
import os
import glob
import numpy as np
from scipy.io import loadmat, savemat
import numpy as np
import scipy.signal as signal
from scipy.signal import cheb2ord
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, cohen_kappa_score, f1_score, precision_score
from collections import OrderedDict
import urllib.request as request
from contextlib import closing
import mne
import warnings
warnings.filterwarnings("ignore")
from scipy.io import loadmat, savemat
import os
import pickle
import csv
from shutil import copyfile
import sys
import resampy
import shutil
import numpy as np
from braindecode.datasets.bbci import BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
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

class LoadData:
    def __init__(self,eeg_file_path: str):
        self.eeg_file_path = eeg_file_path

    def load_raw_data_gdf(self,file_to_load):
        self.raw_eeg_subject = mne.io.read_raw_gdf(self.eeg_file_path + '/' + file_to_load)
        return self

    def load_raw_data_mat(self,file_to_load):
        import scipy.io as sio
        self.raw_eeg_subject = sio.loadmat(self.eeg_file_path + '/' + file_to_load)

    def get_all_files(self,file_path_extension: str =''):
        if file_path_extension:
            return glob.glob(self.eeg_file_path+'/'+file_path_extension)
        return os.listdir(self.eeg_file_path)
class LoadBCIC(LoadData):
    '''Subclass of LoadData for loading BCI Competition IV Dataset 2a'''
    def __init__(self,*args):
        super(LoadBCIC,self).__init__(*args)

    def get_epochs(self, dataPath, labelPath, epochWindow = [0,4], chans = list(range(22))):

        fs = 250
        offset = 2

        # load the gdf file using MNE
        raw_gdf = mne.io.read_raw_gdf(dataPath, stim_channel="auto")
        raw_gdf.load_data()
        event = mne.events_from_annotations(raw_gdf)
        event_id = event[1]['768']
        eventCode = [event_id]
        gdf_events = mne.events_from_annotations(raw_gdf)[0][:, [0, 2]].tolist()
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
        x = x.transpose(2,0,1)

        # Load the labels

        y = loadmat(labelPath)["classlabel"].squeeze()
        # change the labels from [1-4] to [0-3]
        y = y - 1

        data = {'x_data': x, 'y_labels': y, 'c': np.array(raw_gdf.info['ch_names'])[chans].tolist(), 'fs': raw_gdf.info.get('sfreq')}
        return data

class LoadHGD(LoadData):
    '''Subclass of LoadData for loading HGD Dataset'''
    def __init__(self,*args):
        super(LoadHGD,self).__init__(*args)

    def get_epochs(self, dataPath):
        load_sensor_names = None
        low_cut_hz = 4
        fs = 250.0
        loader = BBCIDataset(dataPath, load_sensor_names=load_sensor_names)
        cnt = loader.load()
        marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                                  ('Rest', [3]), ('Feet', [4])])
        clean_ival = [0, 4000]
        set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                             clean_ival)
        clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800
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
        cnt = resample_cnt(cnt, fs)
        cnt = mne_apply(
            lambda a: highpass_cnt(
                a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
            cnt)
        ival = [0, 4000]
        dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
        x = dataset.X[clean_trial_mask]
        y = dataset.y[clean_trial_mask]

        data = {'x_data': x, 'y_labels': y, 'c': C_sensors, 'fs': fs}
        return data

