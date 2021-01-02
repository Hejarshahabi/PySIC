import numpy as np
import pandas as pd
import gdal
import matplotlib.pyplot as plt
import os
class InputData:
    def __init__(self,features, label):
        self.data= gdal.Open(features)
        self.data=self.data.ReadAsArray()
        print("the training dataset has: ")
        print(f'{self.data.shape[0]} Bands, {self.data.shape[1]} Rows, {self.data.shape[2]} Columns')
        print("----------------------------------------------")
        self.inv=gdal.Open(label)
        self.inv=self.inv.ReadAsArray()
        print("the label dataset has: ")
        print(f'{self.inv.shape[0]} Rows, {self.inv.shape[1]} Columns')
        print("----------------------------------------------")
        if self.data.shape[1]!=self.inv.shape[0] or self.data.shape[2]!= self.inv.shape[1]:
            print("the training dataset and label dataset do not have the same shape")
            print ("please reshape the datasets into the same shapes")
            print("----------------------------------------------")
        tempimage= np.zeros((self.data.shape[1]*self.data.shape[2], self.data.shape[0]), dtype="int")
        for i in range(self.data.shape[0]):
            tempimage[:,i]= (self.data[i,:,:]).ravel()
        self.features=tempimage
        print("the shape of new training dataset is: ")
        print(f'{self.features.shape[0]} Rows, {self.features.shape[1]} Column')
        print("----------------------------------------------")
        self.label=self.inv.reshape(self.inv.shape[0]*self.inv.shape[1],1)
        print("the shape of new label dataset is: ")
        print(f'{self.label.shape[0]} Rows, 1 Columns')
        print("----------------------------------------------")

        self.stack=np.hstack((self.features, self.label))
        self.stack=self.stack[self.stack[:,(self.stack.shape[-1]-1)]>0]
    def get_train_data(self):
        return(self.stack[:,:-1], self.stack[:,-1])
    def get_Samples(self):
        self.classes, self.samples = np.unique(self.stack[:,-1], return_counts=True)
        cat=pd.DataFrame({"Class ID": self.classes, 'Samples':self.samples})
        cat.reset_index(drop=True)
        cat.style.hide_index()
        print("ID of each class, and their samples (pixels)")
        print (cat)
        plt.bar(self.classes, self.samples, align="center", alpha=.7)
        plt.xticks(self.classes)
        plt.xlabel("Class ID")
        plt.ylabel("Number of Samples")
        plt.show()
class Plotting:
    def __init__(self, features, labels, figsize=None, hist_bins=None, MarkerSize= None, Marker =None ):
        self.feature=features
        self.labels=labels
        self.figsize = figsize
        self.MarkerSize = MarkerSize
        self.Marker = Marker
        self.hist_bins = hist_bins
        self.title = "Histogram of features and the correlation between classes in each band"
    def plot(self):
        clm=[]
        for i in range(self.feature.shape[1]):
            i=i+1
            clm.append("Band "+str(i))
        self.clm=clm
        self.df=pd.DataFrame(self.feature, columns=clm)
        self.plot = pd.plotting.scatter_matrix(self.df, c=self.labels, figsize=self.figsize, s=self.MarkerSize, marker=self.Marker, hist_kwds={"bins":self.hist_bins})
        plt.suptitle(self.title)
        plt.show()
        return













