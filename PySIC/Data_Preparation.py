import os
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import MinMaxScaler,  StandardScaler
from PySIC.Data import Plotting
import matplotlib.pyplot as plt
import pandas as pd
class Preprocessing:
    def __init__(self,X_Train, Y_Train):
        self.X_train = X_Train
        self.Y_train = Y_Train
    def Scaling(self, scaling=None):
        self.scaling = scaling
        if self.scaling == None or self.scaling == "minmax":
            scale=MinMaxScaler()
            self.Xtrain= scale.fit_transform(self.X_train)
        elif self.scaling == "Standard":
            scale= StandardScaler()
            self.Xtrain=scale.fit_transform(self.X_train)
    def Balancing(self, method=None):
        self.method=method
        if self.method == None or self.method == "nearmiss":
            balance= NearMiss()
            self.Xtrainb, self.Ytrainb =balance.fit_resample(self.Xtrain, self.Y_train)
        elif self.method== "smote":
            balance = SMOTE()
            self.Xtrainb, self.Ytrainb = balance.fit_resample ( self.Xtrain, self.Y_train )
        self.Xtrainb, self.Ytrainb
    def get_balanced_data(self):
        return self.Xtrainb, self.Ytrainb
    def plot(self, figsize=None, hist_bins=None, MarkerSize= None, Marker =None ):
        self.figsize = figsize
        self.MarkerSize = MarkerSize
        self.Marker = Marker
        self.hist_bins = hist_bins
        Plotting(self.Xtrainb, self.Ytrainb, self.figsize,self.hist_bins, self.MarkerSize, self.Marker).plot()
    def get_balanced_samples(self):
        self.classes, self.samples = np.unique ( self.Ytrainb, return_counts=True )
        cat = pd.DataFrame ( {"Class ID" : self.classes, 'Samples' : self.samples} )
        cat.reset_index ( drop=True )
        cat.style.hide_index ( )
        print ( "ID of each class, and their samples (pixels)" )
        print ( cat )
        plt.bar ( self.classes, self.samples, align="center", alpha=.7 )
        plt.xticks ( self.classes )
        plt.xlabel ( "Class ID" )
        plt.ylabel ( "Number of Samples" )
        plt.show ( )

















