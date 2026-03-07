Tenserflow Installation(go to tenserflow website)
Tenserflow is a machine learning library by google for training and deploying deep learning models

after doing all the steps from the tenserflow website for it's installation
1.creating env and activating it
2.installing all requirement.txt which includes all packages as well as tenserflow
3. installing ipykernal
4. remember these tenserflow and all required packages are installed in the tf_env environment of conda only
that's why other python on vs code or other interpreter or kernel will not have tenserflow installed in them 
and they will throw error cuz there is nothing installed about the tenserflow and its packages inside them

5.if require anything for my project so go to that environment(command, cmd) and install anything in them only for the 
project in the same environment

Now environment and all dependencies are installed

Now start working for the project

# important
so if i have installed anaconda for all user then it means it is installed in the program data folder therefor for installing packages with pip or conda i need to go to anaconda as admin privilege but if it was user installation then it could be installed in the user env or jupyter notebook cell - admin(located in program data or folder where i can't write files) , user(inside my user folder env is example)

Music Genre Classifier:

we first set up the tensorflow environment. it is available on the tensorflow website follow the steps and environment is ready(making venv etc everything included)


Data preprocessing 1 - chunks

Data preprocessing 2 - melspectrogram


Data preprocessing 3 - Final
Preparing data and labels for training and testing
  -- Calculate the melspectrogram for all the audio files and save in a variable with its label in other variable



conda list
conda env list
#can use both of them in the cell also
combine with the folder learing 


# Error
function of data preprocessing final:

Inconsistent Shapes: If your dataset has different sampling rates (e.g., some at 44.1kHz and others at 22kHz),
 your chunk_samples and overlap_samples will vary per file.

Resize is your Safety Net: Fortunately, your use of tensorflow.image.resize at the end of the loop will force 
all resulting spectrograms into the (210, 210) shape regardless of the original sampling rate.


pysoundfile error (pysoundfile is a library that audioread uses to read audio files. If it encounters an error, it may be due to a missing dependency or an unsupported audio format.)
audioread acts as a "wrapper." If it doesn't
have a powerful backend like ffmpeg installed on your system, it may fail entirely after the PySoundFile error.

####
installed ffmpeg and saved the zip extract in the folder in the c drive and added the path of the bin folder to the environment variable path and it worked fine after that
Librosa should be able to load your audio files using the audioread backup without issue.


#######
error:
LibsndfileError: Error opening './genres_original\\jazz\\jazz.00054.wav': Format not recognised.
During handling of the above exception, another exception occurred:

    180     warnings.warn(
    181         "PySoundFile failed. Trying audioread instead.", stacklevel=2

Gemini said
This error is a common headache when working with the GTZAN (genres_original) dataset, specifically with the jazz.00054.wav file.
The issue isn't actually your code—it's that the file itself is corrupted or has a malformed header that libsndfile
(the engine behind PySoundFile) can't read.

When libsndfile fails, librosa tries to use audioread as a backup. However, because you are seeing a NoBackendError, it means 
audioread can't find a working decoder (like FFmpeg) to handle the corrupted file.





2. Why FFmpeg didn't save the day yet
You recently installed FFmpeg, but audioread might still be failing because:

Kernel Restart: If you didn't restart your Jupyter Kernel after adding FFmpeg to your System Path,
 the tf_env environment doesn't know it exists yet.

The File is Truly Broken: Sometimes jazz.00054.wav is just a zero-byte file or contains random data
 that no decoder can understand.


summary: 
The Problem: jazz.00054.wav is a known corrupted file in many versions of the GTZAN dataset.

The Cause: PySoundFile crashes, and audioread (even with FFmpeg) can't decode the garbage data in that specific file.

The Solution: Use try-except to skip it so your data and labels arrays can finish building for the other 999 files.
###
Working Solution : skipped the file and it worked fine after that


####
Run from the final data process not the all cells from now on 


### One hot encoding
One-hot encoding is a technique used to convert categorical data into a binary format that can be easily processed by 
machine learning algorithms. In the context of music genre classification, one-hot encoding can be used to represent the
different genres as binary vectors.


##Train test split

##Model Building

Keras(API alows to build and train using below given steps)
1. CNN
2. Models (sequential)
3. Layers (Conv2D, MaxPooling2D, Flatten, Dense)



## Additions of all layers into the sequential model and getting the summary at the last



## Comiling the model

##Training the model
need to tune the model in different ways to increase the accuracy or performance of the model
also tune in a way the training time for(each test and epoches is less)
 
tuning in the..
 1. Sample rate
 2. target_shape (important we reduced from 210*210 to 150*150)
 3. In the layers added(all features and parameters)


saving the model and the training history

training the model is one time process to train it on best parameters and save it for later use


## we can also save the data and lables numpy array by np.save just like model and the training history(why? cause we need to run fun again for data
and labels)


## Model evaluation I
--- on the training data (loss and accuracy)
--- on the test data (loss and accuracy)

--- accuracy and loss validation plots(curves) of (training and validation data)
--- need to tune the model for better accuracy and less loss for validation data

{
  ## the loss and accuracy of the training dataset on the model has a smooth curve
## but for the validation(data- test data) it is somewhat close to the training data but need to be
## improved 

## for improving the accuracy and reducing the loss we need to tune the model as discussed on various
## parameters.

}

## Model evaluation II 
Precision, Recall and Confusion Matrix

- model prediction on X_test gives y_pred
-- need to find pred_categories index value by np.argmax()
-- True categories Y_test also to np.argmax(), max index

-plot confusion matrix on true and predicted categories.
-precision, recall and f1 score by classification report

-Visualization of confusion matrix by heatmap



## Model Prediction for single audio file

1. model load, libraries load
2. audio preprocessing (function which process the audio (gives the X_test))
  --- before feeding the new file to the model we have to preprocess the file in the same format as our model is trained
  it's the same concept of standardization in the linear regression project where we use scaler for the new prediction
  before giving it to the model

3. we will get out X_test which will have chunks in it in(mel's) in np.array format

4. Model prediction function (func which takes X_test)
-- takes the X_test
-- predicts the y_pred(classes in hot encoding)
-- np.argsmax use to find index for each mel
-- find the unique elements and their counts
-- find the unique element having maximum count
-- returning the index of max count
-- accessing class of returned index from classes
