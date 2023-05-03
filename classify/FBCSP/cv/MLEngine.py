import numpy as np
import scipy.signal as signal
from scipy.signal import cheb2ord
from FBCSP import FBCSP
from Classifier import Classifier
import LoadData as LoadData
from sklearn.svm import SVR
import Preprocess as Preprocess
import os
from scipy.io import loadmat, savemat
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, cohen_kappa_score, f1_score, precision_score

class MLEngine:
    def __init__(self,data_path='',file_to_load='',file_to_label='',subject_id='',sessions=[1, 2],ntimes=1,kfold=2,m_filters=2,window_details={}):
        self.data_path = data_path
        self.subject_id=subject_id
        self.file_to_load = file_to_load
        self.file_to_label = file_to_label
        self.sessions = sessions
        self.kfold = kfold
        self.ntimes=ntimes
        self.window_details = window_details
        self.m_filters = m_filters

    def experiment(self):

        # '''for BCIC Dataset'''
        # bcic_data = LoadData.LoadBCIC('1')
        # eeg_data = bcic_data.get_epochs(os.path.join(self.data_path,self.file_to_load),os.path.join(self.data_path,self.file_to_label))

        '''for HGD Dataset'''
        hgd_data = LoadData.LoadHGD('1')
        eeg_data = hgd_data.get_epochs(os.path.join(self.data_path, self.file_to_load))

        fbank = FilterBank(250)
        fbank_coeff = fbank.get_filter_coeff()
        filtered_data = fbank.filter_data(eeg_data.get('x_data'),self.window_details)
        y_labels = eeg_data.get('y_labels')

        training_accuracy = []
        testing_accuracy = []
        testing_precision=[]
        testing_kappa=[]
        testing_f1=[]
        for k in range(self.ntimes):
            '''for N times x K fold CV'''
            # train_indices, test_indices = self.cross_validate_Ntimes_Kfold(y_labels,ifold=k)
            '''for K fold CV by sequential splitting'''
            train_indices, test_indices = self.cross_validate_sequential_split(y_labels)
            '''for one fold in half half split'''
            # train_indices, test_indices = self.cross_validate_half_split(y_labels)
            for i in range(self.kfold):
                train_idx = train_indices.get(i)
                test_idx = test_indices.get(i)
                print(f'Times {str(k)}, Fold {str(i)}\n')
                y_train, y_test = self.split_ydata(y_labels, train_idx, test_idx)
                x_train_fb, x_test_fb = self.split_xdata(filtered_data, train_idx, test_idx)

                y_classes_unique = np.unique(y_train)
                n_classes = len(np.unique(y_train))

                fbcsp = FBCSP(self.m_filters)
                fbcsp.fit(x_train_fb,y_train)
                y_train_predicted = np.zeros((y_train.shape[0], n_classes), dtype=np.float)
                y_test_predicted = np.zeros((y_test.shape[0], n_classes), dtype=np.float)

                for j in range(n_classes):
                    cls_of_interest = y_classes_unique[j]
                    select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]

                    y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
                    y_test_cls = np.asarray(select_class_labels(cls_of_interest, y_test))

                    x_features_train = fbcsp.transform(x_train_fb,class_idx=cls_of_interest)
                    x_features_test = fbcsp.transform(x_test_fb,class_idx=cls_of_interest)

                    classifier_type = SVR(gamma='auto')
                    classifier = Classifier(classifier_type)
                    y_train_predicted[:,j] = classifier.fit(x_features_train,np.asarray(y_train_cls,dtype=np.float))
                    y_test_predicted[:,j] = classifier.predict(x_features_test)


                y_train_predicted_multi = self.get_multi_class_regressed(y_train_predicted)
                y_test_predicted_multi = self.get_multi_class_regressed(y_test_predicted)

                tr_acc =np.sum(y_train_predicted_multi == y_train, dtype=np.float) / len(y_train)
                te_acc =np.sum(y_test_predicted_multi == y_test, dtype=np.float) / len(y_test)

                precision = precision_score(y_test, y_test_predicted_multi, average='macro')
                kappa = cohen_kappa_score(y_test, y_test_predicted_multi)
                f1 = f1_score(y_test, y_test_predicted_multi, average='macro')


                print(f'Training Accuracy = {str(tr_acc)}\n')
                print(f'Testing Accuracy = {str(te_acc)}\n \n')

                training_accuracy.append(tr_acc)
                testing_accuracy.append(te_acc)
                testing_precision.append(precision)
                testing_kappa.append(kappa)
                testing_f1.append(f1)

        mean_training_accuracy = np.mean(np.asarray(training_accuracy))
        mean_testing_accuracy = np.mean(np.asarray(testing_accuracy))
        std_testing_accuracy = np.std(np.asarray(testing_accuracy),ddof=1)
        sem_testing_accuracy = std_testing_accuracy / np.sqrt(len(testing_accuracy))

        mean_testing_precision = np.mean(np.asarray(testing_precision))
        std_testing_precision = np.std(np.asarray(testing_precision), ddof=1)
        sem_testing_precision = std_testing_precision / np.sqrt(len(testing_precision))

        mean_testing_kappa = np.mean(np.asarray(testing_kappa))
        std_testing_kappa = np.std(np.asarray(testing_kappa), ddof=1)
        sem_testing_kappa = std_testing_kappa / np.sqrt(len(testing_kappa))

        mean_testing_f1 = np.mean(np.asarray(testing_f1))
        std_testing_f1 = np.std(np.asarray(testing_f1), ddof=1)
        sem_testing_f1 = std_testing_f1 / np.sqrt(len(testing_f1))

        print('*' * 10, '\n')
        print(testing_accuracy)
        print(testing_kappa)
        print(testing_precision)
        print(testing_f1)
        print('*' * 10, '\n')

        print('*'*10,'\n')
        print(f'Mean Training Accuracy = {str(mean_training_accuracy)}\n')
        print(f'Mean Testing Accuracy = {str(mean_testing_accuracy)}')
        print(f'Std Testing Accuracy = {str(std_testing_accuracy)}')
        print(f'Sem Testing Accuracy = {str(sem_testing_accuracy)}')
        print('*' * 10, '\n')

        print('*' * 10, '\n')
        print(f'Mean Testing precision = {str(mean_testing_precision)}')
        print(f'Std Testing precision = {str(std_testing_precision)}')
        print(f'Sem Testing precision = {str(sem_testing_precision)}')
        print('*' * 10, '\n')

        print('*' * 10, '\n')
        print(f'Mean Testing kappa = {str(mean_testing_kappa)}')
        print(f'Std Testing kappa = {str(std_testing_kappa)}')
        print(f'Sem Testing kappa = {str(sem_testing_kappa)}')
        print('*' * 10, '\n')

        print('*' * 10, '\n')
        print(f'Mean Testing f1 = {str(mean_testing_f1)}')
        print(f'Std Testing f1 = {str(std_testing_f1)}')
        print(f'Sem Testing f1 = {str(sem_testing_f1)}')
        print('*' * 10, '\n')

    def cross_validate_Ntimes_Kfold(self, y_labels, ifold=0):
        from sklearn.model_selection import StratifiedKFold
        train_indices = {}
        test_indices = {}
        random_seed = ifold
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=random_seed)
        i = 0
        for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
            train_indices.update({i: train_idx})
            test_indices.update({i: test_idx})
            i += 1
        return train_indices, test_indices

    def cross_validate_sequential_split(self, y_labels):
        from sklearn.model_selection import StratifiedKFold
        train_indices = {}
        test_indices = {}
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=False)
        i = 0
        for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
            train_indices.update({i: train_idx})
            test_indices.update({i: test_idx})
            i += 1
        return train_indices, test_indices

    def cross_validate_half_split(self, y_labels):
        import math
        unique_classes = np.unique(y_labels)
        all_labels = np.arange(len(y_labels))
        train_idx =np.array([])
        test_idx = np.array([])
        for cls in unique_classes:
            cls_indx = all_labels[np.where(y_labels==cls)]
            if len(train_idx)==0:
                train_idx = cls_indx[:math.ceil(len(cls_indx)/2)]
                test_idx = cls_indx[math.ceil(len(cls_indx)/2):]
            else:
                train_idx=np.append(train_idx,cls_indx[:math.ceil(len(cls_indx)/2)])
                test_idx=np.append(test_idx,cls_indx[math.ceil(len(cls_indx)/2):])

        train_indices = {0:train_idx}
        test_indices = {0:test_idx}

        return train_indices, test_indices

    def split_xdata(self,eeg_data, train_idx, test_idx):
        x_train_fb=np.copy(eeg_data[:,train_idx,:,:])
        x_test_fb=np.copy(eeg_data[:,test_idx,:,:])
        return x_train_fb, x_test_fb

    def split_ydata(self,y_true, train_idx, test_idx):
        y_train = np.copy(y_true[train_idx])
        y_test = np.copy(y_true[test_idx])

        return y_train, y_test

    def get_multi_class_label(self,y_predicted, cls_interest=0):
        y_predict_multi = np.zeros((y_predicted.shape[0]))
        for i in range(y_predicted.shape[0]):
            y_lab = y_predicted[i, :]
            lab_pos = np.where(y_lab == cls_interest)[0]
            if len(lab_pos) == 1:
                y_predict_multi[i] = lab_pos
            elif len(lab_pos > 1):
                y_predict_multi[i] = lab_pos[0]
        return y_predict_multi

    def get_multi_class_regressed(self, y_predicted):
        y_predict_multi = np.asarray([np.argmin(y_predicted[i,:]) for i in range(y_predicted.shape[0])])
        return y_predict_multi


class FilterBank:
    def __init__(self,fs):
        self.fs = fs
        self.f_trans = 2
        self.f_pass = np.arange(4,40,4)
        self.f_width = 4
        self.gpass = 3
        self.gstop = 30
        self.filter_coeff={}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs/2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass+self.f_width])
            f_stop = np.asarray([f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp = f_pass/Nyquist_freq
            ws = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i:{'b':b,'a':a}})

        return self.filter_coeff

    def filter_data(self,eeg_data,window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape
        # if window_details:
        #     n_samples = int(self.fs*(window_details.get('tmax')-window_details.get('tmin')))+1
        filtered_data=np.zeros((len(self.filter_coeff),n_trials,n_channels,n_samples))
        for i, fb in self.filter_coeff.items():
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b,a,eeg_data[j,:,:]) for j in range(n_trials)])
            # if window_details:
            #     eeg_data_filtered = eeg_data_filtered[:,:,int((4.5+window_details.get('tmin'))*self.fs):int((4.5+window_details.get('tmax'))*self.fs)+1]
            filtered_data[i,:,:,:]=eeg_data_filtered

        return filtered_data

