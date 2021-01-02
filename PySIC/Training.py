import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,  StandardScaler
import matplotlib.patches as mpatches


import gdal
import pandas as pd
class Single_Classifier:
    def __init__(self,features,labels, TVsize):
        self.features=features
        self.labels=labels
        self.TVsize=TVsize
        self.X_train, self.X_test, self.y_train, self.y_test= train_test_split(self.features, self.labels, test_size=self.TVsize,random_state=None, stratify=self.labels)
    def Add_Model(self, model=None):
        self.model = model
        if self.model == None:
            self.model = RandomForestClassifier()
        else:
            self.model = model
    def Model_assessment(self, Folds=3):
        self.Folds=Folds
        self.CV = StratifiedKFold(n_splits=self.Folds, random_state=None)
        self.scores = cross_val_score( self.model, self.X_train, self.y_train, cv=self.CV, scoring='accuracy')
    def get_model_score(self):
        print ( f' Accuracy of The Model Based on {self.Folds} Fold Cross Validation:' )
        #print(f'{self.model} >> {self.scores}')
        frame=np.ones((1,self.Folds+1), dtype=object)
        frame[0 , 0] = str ( self.model)
        for i in range (self.Folds):
            frame[0,i+1]=np.round((self.scores[i]),3)
        clm=["Model"]
        for i in range(self.Folds):
            clm.append("Fold "+str(i+1))
        self.df=pd.DataFrame(frame, columns=clm)
        self.df["Mean Accuracy"]=np.round(self.scores.mean(),3)
        print(self.df)
        print("----------------------------------------------------------------------------------------------------------------")
        self.df.to_csv(str(self.model)+"-Training Accuracy.csv")
        print( f"*** Model Accuracy Scores Are Saved as ({str(self.model)}-Training Accuracy.csv) File in Current Directory ***")
        print ( "-------------------------------------------------------------------------------------------------------------" )
        plt.boxplot ( (np.round ( self.scores , 3 )), showmeans=True )
        plt.xticks([1],[str(self.model)])
        plt.title(f"The Trained Model Accuracy Based on {self.Folds} Folds Cross Validation")
        plt.show ( )
    def Model_fitting(self):
        self.model.fit(self.X_train, self.y_train)
    def Model_validation(self):
        self.Y_predicted= self.model.predict(self.X_test)
        cm = confusion_matrix (self.y_test ,  self.Y_predicted)
        disp = ConfusionMatrixDisplay ( confusion_matrix=cm , display_labels=self.model.classes_ )
        disp.plot()
        plt.title("Confusion Matrix")
        plt.show()
        self.cm= cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]
        print("Classification Accuracy for Each Class: ")
        classlist=["Class "+str(i+1) for i in range(len(self.cm.diagonal()))]
        acc=np.zeros((1,len(self.cm.diagonal())))
        for i in range (len(self.cm.diagonal())):
            acc[0,i]=self.cm.diagonal()[i]
        self.accrep=pd.DataFrame(cm, columns=classlist, index=classlist)
        self.accrep["Accuracy"]=np.round(self.cm.diagonal(),3)
        print(self.accrep)
        print('Overall Classification Accuracy: %0.3f' %(self.cm.diagonal().sum()/len(self.cm.diagonal())))
    def save_validation(self):
        if len(self.df.index)==1:
            self.accrep.to_csv(str(self.model)+"-Confusion Matrix.csv")
            print("----------------------------------------------------------------------------------------------------------------")
            print( f"*** Confusion Matrix Scores Are Saved as ({str(self.model)}-Confusion Matrix.csv) File in Current Directory ***")
        else:
            self.accrep.to_csv ( "Meta-model Confusion Matrix.csv" )
            print (f"*** Confusion Matrix Scores Are Saved as (Meta-Model Confusion Matrix.csv) File in Current Directory ***" )
    def Prediction(self, features, scaling=None):
        self.scaling=scaling
        self.new_img= gdal.Open ( features )
        self.new_image = self.new_img.ReadAsArray ( )
        print ( "The Dataset to Be Classified Has: " )
        print ( f'{self.new_image.shape[0]} Bands, {self.new_image.shape[1]} Rows, {self.new_image.shape[2]} Columns' )
        print ( "----------------------------------------------" )
        tempimage = np.zeros ( (self.new_image.shape[1] * self.new_image.shape[2] , self.new_image.shape[0]) , dtype="int" )
        for i in range ( self.new_image.shape[0] ) :
            tempimage[: , i] = (self.new_image[i , : , :]).ravel ( )
        self.new_features= tempimage
        if self.scaling == None or self.scaling == "minmax":
            scale=MinMaxScaler()
            self.feature= scale.fit_transform(self.new_features)
        elif self.scaling == "Standard":
            scale= StandardScaler()
            self.feature=scale.fit_transform(self.new_features)

        self.prediction=self.model.predict(self.feature)
        print("Classification Is Done")
        self.map= self.prediction.reshape(self.new_image.shape[1],self.new_image.shape[2])
    def get_ClassificationMap(self):
        return self.map
    def PlotMap(self):
        self.cls, self.nump= np.unique(self.map, return_counts=True)
        self.clslist=[]
        for i in range(len(self.cls)):
            self.clslist.append("Class: "+str(i+1))
        plt.figure(figsize=(10,10))
        img=plt.imshow(self.map, interpolation=None)
        colors = [img.cmap ( img.norm ( cls ) ) for cls in self.cls]
        patches = [mpatches.Patch ( color=colors[i] , label=self.clslist[i] ) for i in range ( len ( self.cls ) )]
        plt.legend ( handles=patches , bbox_to_anchor=(1.07, 1) , loc= 2 , borderaxespad=0., fontsize=12, title= "Class ID" )
        plt.title("Classification Map", fontsize=16 )
        plt.show()
    def ExprotMap(self, MapName):
        self.mapname=MapName
        driver=gdal.GetDriverByName("GTiff")
        driver.Register()
        output=driver.Create(self.mapname, xsize=self.map.shape[1], ysize=self.map.shape[0], bands=1, eType=gdal.GDT_Int16)
        output.SetGeoTransform(self.new_img.GetGeoTransform())
        output.SetProjection(self.new_img.GetProjection())
        outputband=output.GetRasterBand(1)
        outputband.WriteArray(self.map)
        outputband.SetNoDataValue(0)
        outputband.FlushCache()
        outputband= None
        output= None
        print( f"Classification Map is Sotred as '{str(self.mapname)}' in Current Directory ")
class Multi_Classifier(Single_Classifier):
    def __init__(self,features, labels, TVsize):
        super().__init__(features, labels, TVsize)
    def Base_models(self, base_models=None):
        self.base = base_models
        if self.base == None:
            self.base = list()
            self.base.append(("RF", RandomForestClassifier(n_estimators=100)))
            self.base.append(("SVM", SVC()))
            self.base.append(("MLP", MLPClassifier(max_iter=500)))
        else:
            self.base= base_models
        print("----------------------------------------------------------------------------------------------------------------")
        print("Base Models Are as Below:")
        print (pd.DataFrame(self.base,columns=["Model Name", "Model Parameters"]))
        print('----------------------------------------------------------------------------------------------------------------')
    def Base_model_assessment(self, Folds=3):
        #self.classifier=self.base
        self.Folds=Folds
        self.modell, self.accuracy = list(), list()
        for self.name, self.model in (dict(self.base)).items():
            Single_Classifier.Model_assessment(self, self.Folds)
            self.accuracy.append((self.scores))
            self.modell.append(self.name)
    def get_accuracy(self):
        print(f' Accuracy of Each Model (base model) based on {self.Folds} Fold Cross Validation:')
        #print(f'{self.model} >> {self.accuracy}')
        r= np.zeros((len(self.base),self.Folds+1))
        accuracy=np.array(np.round(self.accuracy,3))
        r = np.ones ( (r.shape), dtype=object )
        self.mscores=np.ones((len(self.base),1))
        for i in range ( len ( self.base ) ) :
            for j in range ( self.Folds ) :
                r[i, 0] = self.base[i]
                r[i, j + 1] = accuracy[i, j]
                self.mscores[i,0]=accuracy[i,:].mean()
        column=["Base Model"]
        for i in range(self.Folds):
            column.append("Fold "+ str(i+1))
        df=pd.DataFrame(r , columns=[column])
        df["Mean"]=np.round(self.mscores,3)
        df.to_csv("Base-Models Training Accuracy.csv")
        print(df)
        print("----------------------------------------------------------------------------------------------------------------")
        print(f"*** Model Accuracy Scores Are Saved as Base-Model Training Accuracy.csv File in Current Directory ***" )
        print("----------------------------------------------------------------------------------------------------------------")
        plt.boxplot((self.accuracy), labels=self.base, showmeans=True )
        plt.title(f"Base Models Accuracy Based on {self.Folds} Fold Cross  Validation")
        plt.show ( )
    def Meta_model(self, meta_model=None, CV=None):
        self.model = meta_model
        if self.model == None:
            self.model= LogisticRegression()
        else:
            self.model= meta_model
        self.cv=CV
        if self.cv==None:
            self.cv=5
        else:
            self.cv=CV
    def Meta_model_assessment(self):
        self.stack=StackingClassifier(self.base, self.model, cv=self.cv)
        Single_Classifier.Model_assessment (self, Folds=self.cv)
        list1=["Fold "+str(i+1) for i in range(self.cv)]
        list=["Meta Model"]
        for i in list1:
            list.append(i)
        df=np.ones((1, self.cv+1), dtype=object)
        df[0,0]=str(self.model)
        for i in range(self.cv):
            df[0,i+1]=np.round(self.scores[i],3)
        df=pd.DataFrame(df, columns=list)
        df["Mean"]=np.round(self.scores.mean(),3)
        print(f" Accuracy of The Meta-Model Based on {self.cv} Fold Cross Validation:")
        print(df)
        print("----------------------------------------------------------------------------------------------------------------")
        plt.boxplot( (np.round(self.scores,3)) ,showmeans=True)
        plt.xticks ( [1] , ["Metal Model"] )
        plt.title(f"Meta Model Accuracy Based on {self.cv} Fold Cross Validation")
        plt.show ( )
    # def Model_fitting(self):
    #     self.model=self.stack
    #     self.model.fit(self.X_train, self.y_train)
    # def Model_validation(self):
    #     Single_Classifier.Model_validation(self)
    # def save_validation(self):
    #     Single_Classifier.save_validation(self)
    def save_validation(self):
        print("----------------------------------------------------------------------------------------------------------------")
        self.accrep.to_csv ( "Meta-model Confusion Matrix.csv" )
        print (f"*** Confusion Matrix Scores Are Saved as (Meta-Model Confusion Matrix.csv) File in Current Directory ***" )











