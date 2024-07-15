import numpy as np
import pandas as pd
import vitaldb as vf
from os import listdir
from scipy import ndimage
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

from modules.filter import outlierfilter

class DatasetImport():
    def __init__(self, directory: str, dataset: str, vitalpath: str, interval: int = 10):
        self.directory = directory
        self.datasetpath = directory + dataset
        self.vitalpath = directory + vitalpath

        self.interval = interval

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        self.index = pd.read_csv(self.datasetpath +'dataset.csv', index_col=0).index.to_numpy()

    def save(self, filename: str):
        np.savez_compressed(self.datasetpath+filename,
                            train = self.train_dataset,
                            validation = self.validation_dataset,
                            test = self.test_dataset,
                            timesteps = self.timesteps,
                            )

    def load(self, filename: str):
        data = np.load(self.datasetpath+filename)
        self.train_dataset = data['train']
        self.validation_dataset = data['validation']
        self.test_dataset = data['test']
        try:
            self.timesteps = data['timesteps']
        except:
            self.timesteps = []

    def split(self,data):
       train, test = train_test_split(data, test_size=0.15, random_state=42)
       train, validation = train_test_split(train, test_size=0.15, random_state=42)
       return train, validation, test

    def generateDataset(self, normalization):

        dataset, self.timesteps = self.generate(self.index, normalization)

        self.train_dataset, self.validation_dataset, self.test_dataset = self.split(dataset)
        print('Dataset succesfully generated                 ')

    def generate(self, dataset_index: list, normalization):
        batch_list = []
        timesteps = []

        for i, caseid in enumerate(dataset_index):
            filepath = self.vitalpath+str(caseid).zfill(4)+'.vital'
            data, importName = self.importFunction(filepath)
            timesteps.append(data.shape[0])
            batch_list.append(data)
            print(importName + " Fortschritt: %.1f" % (100 * (i+1) / len(dataset_index)),' % ', end='\r')

        ### Pad the dataset
        data = pad_sequences(batch_list, padding='post', dtype='float32', value=0.0)

        # Remove 0.0 padded values
        data[data == 0.0] = np.nan

        # Nomalization
        data = normalization(data)

        # restore padded values
        np.nan_to_num(data, copy=False, nan=0.0)

        return data, np.array(timesteps)

    def importFunction(self, filepath: str):
        return None, None

class infoImport(DatasetImport):
    def __init__(self, directory: str, dataset: str, vitalpath: str):
        super().__init__(directory,dataset,vitalpath)

        self.columns = ['sex','age','height','weight','bmi']

    def generate(self, dataset_index: list, normalization):

        data = pd.read_csv(self.directory+'info_vitaldb/cases.csv', index_col=0)
        data = data[self.columns].loc[dataset_index].to_numpy()

        sex = np.where(data[:, 0] == 'F', -0.5, 0.5)

        data = data[:,1:].astype(float)
        data = np.c_[sex, normalization(data)]

        return data, None

class VitalImport(DatasetImport):
    def __init__(self, directory: str, dataset: str, vitalpath: str):
        super().__init__(directory,dataset,vitalpath)

        self.tracks = []
        self.filter = [0,0,0]
        self.name = 'Vital'

    def importFunction(self, filepath: str):

        vitaldata = vf.VitalFile(ipath = filepath, track_names = self.tracks)

        data = vitaldata.to_pandas(track_names=self.tracks,interval=self.interval)
        data = data + 0.00001 # adds small value to avoid mix up with padding values
        data = outlierfilter(data, threshhold = self.filter[0] , iterations = 2, min = self.filter[1], max = self.filter[2])

        return data, self.name

class BPImport(DatasetImport):
    def __init__(self, directory: str, dataset: str, vitalpath: str):
        super().__init__(directory,dataset,vitalpath)

    def importFunction(self, filepath: str):
        pressureWave = vf.VitalFile(filepath).to_numpy(['SNUADC/ART'], 1/500)

        samples = self.interval * 500

        # Remove values which derivative is too large
        gradient = np.diff(pressureWave,n=1, axis=0, append=0)
        gradientfilter1 = ndimage.binary_dilation(np.abs(gradient) > 4,iterations=30)
        gradientfilter2 = ndimage.binary_dilation(np.abs(gradient) > 7,iterations=1000)
        pressureWave[gradientfilter1] = np.nan
        pressureWave[gradientfilter2] = np.nan

        # Remove the negative values and values above 250
        pressureWave[pressureWave <= 20] = np.nan
        pressureWave[pressureWave > 250] = np.nan

        pressureWave = self.imputer1.fit_transform(pressureWave)

        ### Reshape the pressureWave to 1000 samples (2 seconds) per row
        #if (pressureWave.shape[0] % samples) != 0 :
        #    steps2fill = samples - (pressureWave.shape[0] % samples)
        #    pressureWave = np.pad(array=pressureWave, pad_width=((0,steps2fill),(0,0)), mode='constant', constant_values=np.nan)
        length = pressureWave.shape[0] - (pressureWave.shape[0] % samples)
        pressureWave = pressureWave[0:length]
        return pressureWave.reshape(-1,samples), 'Blood Pressure'