import numpy as np
from scipy.io import loadmat
import random
from pylab import *
from numpy import *
from scipy import interpolate
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import cohen_kappa_score
import random

class DataProcess():
    '''
    load data from BCI Competition 2a and 2b
    '''
    data = []
    label = []
    
    '''
    data_path: the data files path
    data_files:the data files name(ex:["A01T"] or ["B0103E"])
    choose2aor2b: if choose 2a dataset set 1;else set 2
    choose2aclasses: if choose all 4 classes in 2a dataset set 1;
                     if choose left hand and right hand 2 classes set 2;
                     if choose foot and tongue 2 classes set 3.
    
    '''
    def __init__(self,
                data_path,
                data_files,
                choose2aor2b,
                choose2aclasses = None):
        self.data_path = data_path
        self.data_files = data_files
        self.choose2aclasses = choose2aclasses
        self.choose2aor2b = choose2aor2b
        
        if choose2aor2b == 1:
            '''
            choose dataset 2a
            '''
            data, label = self.load_npy_for_2a(self.data_files,self.data_path,choose2aclasses)
            data = self.interpolate_data(data,orig = 1000,porlong = 1050,kind = "cubic")

        elif choose2aor2b == 2:
            '''
            choose dataset 2b
            '''
            data,label = self.load_npy_for_2b(self.data_files,self.data_path)
            
        # Normalized
        data = data.swapaxes(1,2)
        data -= data.mean(axis=0)
        data /= data.std(axis=0)
        data = data.swapaxes(1,2)

        # Shuffle
        index_k = [i for i in range(len(data))] 
        random.shuffle(index_k)
        self.data = data[index_k]
        self.label = label[index_k]
        print(self.data.shape)
        print(self.label.shape)
                
    '''
    Sliding window data augmentation
    '''
    def data_augmentation(self,data,label,windows_long,interval):
        new_group = [] 
        new_label = []
        for j in range(len(data)):
            data_t = data[j]                
            label_t = label[j]
            data_t3 = data_t.T 
            limit = data_t3.shape[1]
            windows_plus = windows_long
            start = 0
            while windows_plus<limit:
                new_t = []
                for i in range(data.shape[2]): # deal with all channels 
                    one_cha = data_t3[i]    
                    cut = one_cha[start:windows_plus]
                    cut = np.array(cut)
                    new_t.append(cut)
                new_t = np.array(new_t)
                new_t = new_t.T
                new_group.append(new_t)
                new_label.append(label_t)
                windows_plus += interval
                start += interval
        new_group = np.array(new_group)
        new_label = np.array(new_label)
        return new_group,new_label
    
    '''
    Gauss data augmentation
    m:multiple
    '''
    def gauss_data_augmentation(self,data,label,sigma,m = 1):
        gauss_data = data
        gauss_label = label
        if m == 1:
            return gauss_data,gauss_label
        for k in range(1,m):
            new_data = []
#             print(data.shape)
            for i in range(0,data.shape[2]):
                ch = data[:,:,i]
                new_ch = []
                for j in range(len(ch)):
                    pch = ch[j] + random.gauss(0,sigma)
                    new_ch.append(pch)
                new_ch = np.array(new_ch)
                new_data.append(new_ch)
            new_data = np.array(new_data)
            new_data = new_data.T
            new_data = new_data.swapaxes(0,1) 
            gauss_data = np.concatenate((gauss_data,new_data),axis = 0)
            gauss_label = np.concatenate((gauss_label,label),axis = 0)
        return gauss_data,gauss_label
    
    '''
    Interpolate data
    'zero' 'nearest' stepped interpolation
    'slinear' 'linear'  Linear interpolation
    'quadratic' 'cubic' 2 or 3 order B-spline curves
    
    orig: the original data length
    prolong: the final data length
    kind: interpolate type
    '''
    def interpolate_data(self,datat,orig = 350,porlong = 1050,kind = "quadratic"):
        datat = datat.swapaxes(1,2)
        new_data = []
        for i in datat:
            new_channel = []
            for k in i:
                x = np.linspace(0,porlong,orig)     
                f = interpolate.interp1d(x,k,kind = "quadratic")  
                xnew = np.linspace(0,porlong,porlong) 
                ynew = f(xnew)                 
                new_channel.append(ynew)
            new_data.append(new_channel)
        new_data = np.array(new_data)
        new_data = new_data.swapaxes(1,2)
        return new_data
    
    '''
    load 2b dataset
    '''
    def load_npy_for_2b(self,data_files,data_path):
        data = []
        label = []
        for i in range(len(data_files)):
            data_file = data_path + data_files[i] + "_data_raw.npy"
            label_file = data_path + data_files[i] + "_label.npy"
            if i == 0:
                data = np.load(data_file)
                label = np.load(label_file)
            else:
                data_t = np.load(data_file)
                label_t = np.load(label_file)
                data = np.concatenate((data,data_t),axis = 0)
                label = np.concatenate((label,label_t),axis = 0)
            print(data_file[-19:-4],"load success.")
        return data,label
    
    '''
    choose 2a classes
    '''
    def choose_2a_class(self,cnt_data,cnt_label,choose2aclasses):
        classes = cnt_label
        left = []
        right = []
        foot = []
        tongue = []
        for i in range(len(classes)):
            if classes[i] == 1:
                left.append(i)
            elif classes[i] == 2:
                right.append(i)
            elif classes[i] == 3:
                foot.append(i)
            else:
                tongue.append(i)
        left = np.array(left)
        right = np.array(right)
        foot = np.array(foot)
        tongue = np.array(tongue)

        # data
        data_left = []
        data_right = []
        data_foot = []
        data_tongue = []
        for i in range(len(cnt_data)):
            if i in left:
                data_left.append(cnt_data[i])
            elif i in right:
                data_right.append(cnt_data[i])
            elif i in foot:
                data_foot.append(cnt_data[i])
            else:
                data_tongue.append(cnt_data[i])
        data_left = np.array(data_left)
        data_right = np.array(data_right)
        data_foot = np.array(data_foot)
        data_tongue = np.array(data_tongue)

        label_left = [1 for i in range(len(data_left))]
        label_right = [2 for i in range(len(data_right))]
        label_foot = [3 for i in range(len(data_foot))]
        label_tongue = [4 for i in range(len(data_tongue))]
        if choose2aclasses == 2:
            cnt_data = np.concatenate((data_left,data_right),axis = 0)
            cnt_label = np.concatenate((label_left,label_right),axis = 0)
        elif choose2aclasses == 3:
            cnt_data = np.concatenate((data_foot,data_tongue),axis = 0)
            cnt_label = np.concatenate((label_foot,label_tongue),axis = 0)
        cnt_data = np.array(cnt_data)
        cnt_label = np.array(cnt_label)

        index = [i for i in range(len(cnt_data))] 
        random.shuffle(index)
        cnt_data = cnt_data[index]
        cnt_label = cnt_label[index]
        return cnt_data,cnt_label

    '''
    load 2a dataset
    '''
    def load_npy_for_2a(self,raw_gdf,file_path,choose2aclasses):
        data = []
        label = []
        for i in range(len(raw_gdf)):
            data_path = self.data_path + raw_gdf[i] + "_data.npy"
            label_path = self.data_path + raw_gdf[i] + "_label.npy"
            if i == 0:
                data = np.load(data_path)
                data = data.swapaxes(0,1)
                data = data.swapaxes(1,2)
                label = np.load(label_path)
            else:
                data_t = np.load(data_path)
                data_t = data_t.swapaxes(0,1)
                data_t = data_t.swapaxes(1,2)
                label_t = np.load(label_path)
                data = np.concatenate((data,data_t),axis = 0)
                label = np.concatenate((label,label_t),axis = 0)
            print(data_path[-13:-9],"load success.")
        data,label = self.choose_2a_class(data,label,choose2aclasses)

        data -= data.mean(axis=0)
        data /= data.std(axis=0)
        # one-hot 
        one_hot_label = array(label)
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(one_hot_label)
        onehot_encoder = OneHotEncoder(sparse = False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        label = np.array(onehot_encoded)
        return data,label
