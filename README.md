# PySIC
**PySIC** stands for Python Satellite Imagery Classifier is a powerful
and easy to implement package for satellite image classification using
machine learning (ML) model. All available ML models in Scikit-Learn
library can be used in PySIC as a classifier model, in addition to
single classifier model, another ensemble ML model called Stacking is
used to combine multiple ML models together to build a powerful
classifier model. More information on Stacking model can be found in
on [Scikit-Learn website](https://scikit-learn.org/stable) and in this [paper](https://www.mdpi.com/1424-8220/19/22/4893) by Shahabi et al, 2019.

![enter image description here](https://www.mdpi.com/sensors/sensors-19-04893/article_deploy/html/images/sensors-19-04893-g005-550.jpg)

**Shahabi, Hejar, et al**. "A semi-automated object-based gully networks detection using different machine learning models: A case study of Bowen Catchment, Queensland, Australia." _Sensors_ 19.22 (2019): 4893.

For model assessment and validation, Stratified K-Fold is used to
evaluate the performance of the selected model, also, the training data
set is divided in two sets for model assessment and model validation.
The ration of the division is set by user. **PySIC** gets features ( can
be a stack of satellite images, NDVI index, Slope layer and..) and
labels as **.tif** files as inputs and its output is a classification map
with .tif format. In this package, before applying classification, some
pre-processing steps including data scaling, standardizing and balancing
are taken to reduce the inconsistency and imbalance among features and
classes, respectively. The package is built on the libraries such as
**Numpy**, **Pandas**, **Matplotlib**, **Scikit-Learn**, **imbalanced-learn**, and **GDAL**,
therefore installing these libraries are required. 
Please do not use this package for commercial purpose without my explicit permission.
Researchers/ academicians are welcomed for feedback and technical
support. 
# PySIC Requirements:

 1. **Numpy**
 2. **Pandas**
 3. **Scikit-Learn**
 4. **Imbalanced-learn**
 5. **GDAL**
 5. **Matplotlib**

 The version of **GDAL** should matches your python version,using the following [link](https://www.lfd.uci.edu/~gohlke/pythonlibs/) you can download **GDAL** file that matches your python version.  
To install **GDAL** manually follow these steps: First download it in your local drive, then in your terminal environment type this code:  
pip install C:\......\GDAL3.x.x‘cpx‘cpx‘win_amd64.whl.

# A Guide on how to use this package

    import os

    os.chdir('your data directory')

*To load data Data module should be called*

    from PySIC import Data

*The first and second arguments are features, and label raster, respectively.Invenotry or label values should start from 1 not zero, so make sure that 0 is not associated with any classes in inventory layer. It returns information on data bands, rows and columns and then convert them to a 2D matrices. Each column in new training dataset represents an image or feature band, and each row shows a pixel*

    instance=Data.InputData(data,inventory)

*with this code you can get reshaped training features and labels*

    features, labels=instance.get_train_data()


*with this code you can get the labels and the number of samples(pixels)associated with each label or class as well as the graph. your samples might be imbalanced, but in following you'll learn how to balanced them using different methods*

    instance.get_Samples()

*this code takes features and labels as inputs to visualize them as below this visualization can help to evaluate the distribution of your classes based on the features Diagonal histograms are features or image bands*

    plot=Data.Plotting(features,labels,figsize=[10,10], hist_bins=50, MarkerSize=10, Marker=".")
    plot.plot()
## Data Preparation for classification

    from PySIC import Data_Preparation

*To preprocess features and labels for classification*  

    data=Data_Preparation.Preprocessing(features, labels)

## Scaling
S*ince machine learning models (except random forest and decision tree)
are based on calculating distances, all features should be scaled to the
same range here to scaling methods called MinMax and StandardScalar,
which are available in Scikit-Learn package are used the following code
is used for scaling, and accepts string as input. if you input "minmax"
it uses MinMax method or if you use standard it uses "StandardScalar" if
you put nothing it uses the default method which is MinMax method.*  

    data.Scaling("minmax")

## DataBalance
*To make a balance between(under-sampling/over-sampling) classes and their samples, the following code is used. It gets string input such as"nearmiss" or "smote" to balance data. Default method is "NearMiss" more information on these methods please visit [nearmiss](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.under_sampling.NearMiss.html) and [smote](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html) documentation webpage.

    data.Balancing()

*The following code shows the balanced samples based on the method you applied previously.*  

    data.get_balanced_samples()

*The following code shows the balanced samples based on the method you applied previously.*  

    data.get_balanced_samples()

*This code plots balanced features and labels.*  

    data.plot(figsize=[10,10], hist_bins=50, MarkerSize= 10, Marker=".")
## Model Training 
*In this section to approaches are used for image classification.
First method is using single classifier and second method is stacking mutiple classfier together as one powerful model.*

 ## USing Single Classifier
*Inputs are **X** and **Y** that we got from last code.  
the third argument (**0.3**) is the ratio between training data and validation data*  

    single=Training.Single_Classifier(X,Y,0.3)

*Here you have to introduce your own model wit its parameters for example:*

    mymodel= MLPClassifier(hidden_layer_sizes=200,activation='relu',max_iter=500 )   
*Using this code you add your defined model.*

    single.Add_Model(mymodel)

*To evaluate model performance "StratifiedKFold" for cross validation. the following code gets an integer as splits or folds to split data for model assessment.*  

    single.Model_assessment(Folds=5)

*Using following code, model accuracy in different fold as well as the mean accuracy are shown the accuracy report is stored in current directory.*  

    single.get_model_score()

**Fitting model**  

      single.Model_fitting()

*this code validate fitted model on some unseen data(validation data).*

    single.Model_validation()

*In order to apply the train model on your new data set you can use this code it takes to input frist is your image or features and the second one is the scaling method that was used to scale the training data. 

    single.Prediction("test2.tif", "minmax")

 ## Stacking Classifier

    from PySIC import Training
    stacking= Training.Multi_Classifier(X,Y,0.3)

*using this code base models can be introduced. Here in our example we just use four base models: **RF**, **SVM**, **Dtree** and **MLP**.*  
*default* `stacking.Base_models()` *base models are **RF**, **MLP** and **SVM**.*

*the following format should be used in intordducing your own base models:*

    baseModel = (("RF",RandomForestClassifier(n_estimators=100)),("SVM", SVC()),("Dtree", DecisionTreeClassifier()),("MLP", MLPClassifier(hidden_layer_sizes=200)))
  **Adding* base models*

    stacking.Base_models(baseModel)

*like single classifer, number of folds must be *added* for base models* assessment.  

    stacking.Base_model_assessment(Folds=4)

*This code returns each base model training accuracy in each fold.*  

    stacking.get_accuracy()

with this code you should introduce your meta classifier and number for cross validation folds default meta model is `LogesticRegression()` (it is recommended to use the default model) and **CV** is 5 this code can be passed with no inputs. but here we used **MLP** as our meta model.

    stacking.Meta_model(MLPClassifier(), CV=4)

*To evaluate the meta model performance*

    stacking.Meta_model_assessment()

**Fitting model**

    stacking.Model_fitting()

*Model validation*

    stacking.Model_validation()

*Save validation data as a **.csv** file in current directory*  

    stacking.save_validation()

**Making predictions.**  

    stacking.Prediction("test2.tif", scaling="minmax")

*This code plot the map*

    stacking.PlotMap()
*To save the classification map in tif format following code can be use.
takes the a name for the map and stores in the current directory.*  

    stacking.ExprotMap("prediction.tif")














































