
# Artificial Intelligence Nanodegree

## Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App 

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this IPython notebook.



---
### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 6](#step6): Write your Algorithm
* [Step 7](#step7): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

### Import Dog Dataset

In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
- `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
- `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels 
- `dog_names` - list of string-valued dog breed names for translating labels


```python
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
```

    Using TensorFlow backend.
    

    There are 133 total dog categories.
    There are 8351 total dog images.
    
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.
    

### Import Human Dataset

In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.


```python
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))
```

    There are 13234 total human images.
    

---
<a id='step1'></a>
## Step 1: Detect Humans

We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.

In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1
    


![png](./images/output_5_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ 


```python
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.

def face_detector_performance_test(image_array):
    count = 0
    total = len(image_array)
    for img in image_array:
        count += face_detector(img)
    return count/total*100

print('The accuracy of detecting human face in "human_files_short" is %.2f%%.' % face_detector_performance_test(human_files_short) )
print('The accuracy of detecting human face in "dog_files_short" is %.2f%%.' % face_detector_performance_test(dog_files_short) )
```

    The accuracy of detecting human face in "human_files_short" is 99.00%.
    The accuracy of detecting human face in "dog_files_short" is 11.00%.
    

__Question 2:__ This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

__Answer:__

We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on each of the datasets.

__My answer:__

Yes. Think about twins, they looks very close to each other when they are in infant. As time passes, there will be slightly different between them on the face. However, it is high possible that they still look very close to each other when they are laughing or crying since the slightly different things on the face will be diminished. That's why I think it is a reasonable expectation to pose on the user to provide a clear view of a face.


```python
## (Optional) TODO: Report the performance of another  
## face detection algorithm on the LFW dataset
### Feel free to use as many code cells as needed.
```

---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.


```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```

### Pre-process the Data

When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

$$
(\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),
$$

where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  

The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape

$$
(1, 224, 224, 3).
$$

The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape 

$$
(\text{nb_samples}, 224, 224, 3).
$$

Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!


```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### Making Predictions with ResNet-50

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing.  First, the RGB image is converted to BGR by reordering the channels.  All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as $[103.939, 116.779, 123.68]$ and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.  This is implemented in the imported function `preprocess_input`.  If you're curious, you can check the code for `preprocess_input` [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).

Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions.  This is accomplished with the `predict` method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category.  This is implemented in the `ResNet50_predict_labels` function below.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a). 


```python
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

### Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).

We use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 3:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ 


```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

def dog_detector_performance_test(image_array):
    count = 0
    total = len(image_array)
    for img in image_array:
        count += dog_detector(img)
    return count/total*100

print('The accuracy of detecting dog in "human_files_short" is %.2f%%.' % dog_detector_performance_test(human_files_short) )
print('The accuracy of detecting dog in "dog_files_short" is %.2f%%.' % dog_detector_performance_test(dog_files_short) )
```

    The accuracy of detecting dog in "human_files_short" is 2.00%.
    The accuracy of detecting dog in "dog_files_short" is 100.00%.
    

---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 1%.  In Step 5 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

Be careful with adding too many trainable layers!  More parameters means longer training, which means you are more likely to need a GPU to accelerate the training process.  Thankfully, Keras provides a handy estimate of the time that each epoch is likely to take; you can extrapolate this estimate to figure out how long it will take for your algorithm to train. 

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun! 

### Pre-process the Data

We rescale the images by dividing every pixel in every image by 255.


```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

    100%|██████████| 6680/6680 [00:53<00:00, 125.47it/s]
    100%|██████████| 835/835 [00:05<00:00, 139.68it/s]
    100%|██████████| 836/836 [00:06<00:00, 139.07it/s]
    

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        model.summary()

We have imported some Python modules to get you started, but feel free to import as many modules as you need.  If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:

![Sample CNN](images/sample_cnn.png)
           
__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

__Answer:__ 

__My answer:__
I started from the same architecture above but changed the few last steps.

Instead of using "global average pooling" layer, my architecture is using "Flatten" layer followed by two dense layer "relu" and "softmax".

This is the one I was using before and it was doing a great job on predicting the object.

However, from the result below you will notice it has larger parameters to tune than the architecture above has.

Since the latter one has less parameters, it should result in a faster computation, better result(may be a slightly worse but acceptable?).

This will need to be tested and verified.

My architecture is following:
___

INPUT

CONV

POOL

CONV

POOL

CONV

POOL

FLATTEN

DENSE

DENSE

OUTPUT
___


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: Define your architecture.

# Use 16 filters
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=train_tensors.shape[1:]))
# Use max pooling with size 2
model.add(MaxPooling2D(pool_size=2))
# Use 32 filters
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
# Use max pooling with size 2
model.add(MaxPooling2D(pool_size=2))
# Use 64 filters
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
# Use max pooling with size 2
model.add(MaxPooling2D(pool_size=2))
# Flatten into 1 dimension vector
model.add(Flatten())
# Use RELU function in the flatten layer
model.add(Dense(500, activation='relu'))
# Return probabilities
model.add(Dense(133, activation='softmax'))

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 224, 224, 16)      208       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 112, 112, 16)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 112, 112, 32)      2080      
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 56, 56, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 56, 56, 64)        8256      
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 28, 28, 64)        0         
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 50176)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 500)               25088500  
    _________________________________________________________________
    dense_2 (Dense)              (None, 133)               66633     
    =================================================================
    Total params: 25,165,677.0
    Trainable params: 25,165,677.0
    Non-trainable params: 0.0
    _________________________________________________________________
    

### Compile the Model


```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 5

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/5
    6660/6680 [============================>.] - ETA: 0s - loss: 5.0295 - acc: 0.0186Epoch 00000: val_loss improved from inf to 4.58208, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 33s - loss: 5.0283 - acc: 0.0187 - val_loss: 4.5821 - val_acc: 0.0371
    Epoch 2/5
    6660/6680 [============================>.] - ETA: 0s - loss: 4.1056 - acc: 0.1023Epoch 00001: val_loss improved from 4.58208 to 4.23394, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 32s - loss: 4.1050 - acc: 0.1024 - val_loss: 4.2339 - val_acc: 0.0766
    Epoch 3/5
    6660/6680 [============================>.] - ETA: 0s - loss: 2.5012 - acc: 0.4093Epoch 00002: val_loss did not improve
    6680/6680 [==============================] - 31s - loss: 2.5011 - acc: 0.4093 - val_loss: 5.1103 - val_acc: 0.0719
    Epoch 4/5
    6660/6680 [============================>.] - ETA: 0s - loss: 0.7250 - acc: 0.8206Epoch 00003: val_loss did not improve
    6680/6680 [==============================] - 31s - loss: 0.7259 - acc: 0.8201 - val_loss: 7.3037 - val_acc: 0.0719
    Epoch 5/5
    6660/6680 [============================>.] - ETA: 0s - loss: 0.1601 - acc: 0.9637Epoch 00004: val_loss did not improve
    6680/6680 [==============================] - 31s - loss: 0.1597 - acc: 0.9638 - val_loss: 8.4384 - val_acc: 0.0647
    




    <keras.callbacks.History at 0x7fa1a01c4d30>



### Load the Model with the Best Validation Loss


```python
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### Test the Model

Try out your model on the test dataset of dog images.  Ensure that your test accuracy is greater than 1%.


```python
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 7.1770%
    

---
<a id='step4'></a>
## Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning.  In the following step, you will get a chance to use transfer learning to train your own CNN.

### Obtain Bottleneck Features


```python
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

### Model Architecture

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.


```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_1 ( (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229.0
    Trainable params: 68,229.0
    Non-trainable params: 0.0
    _________________________________________________________________
    

### Compile the Model


```python
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### Train the Model


```python
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/20
    6440/6680 [===========================>..] - ETA: 0s - loss: 12.0434 - acc: 0.1205Epoch 00000: val_loss improved from inf to 10.58278, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 11.9649 - acc: 0.1250 - val_loss: 10.5828 - val_acc: 0.2048
    Epoch 2/20
    6460/6680 [============================>.] - ETA: 0s - loss: 9.7368 - acc: 0.2944Epoch 00001: val_loss improved from 10.58278 to 9.92005, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 9.7589 - acc: 0.2942 - val_loss: 9.9200 - val_acc: 0.2635
    Epoch 3/20
    6480/6680 [============================>.] - ETA: 0s - loss: 9.2585 - acc: 0.3554Epoch 00002: val_loss improved from 9.92005 to 9.70240, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 9.2486 - acc: 0.3558 - val_loss: 9.7024 - val_acc: 0.2862
    Epoch 4/20
    6420/6680 [===========================>..] - ETA: 0s - loss: 9.0276 - acc: 0.3941Epoch 00003: val_loss improved from 9.70240 to 9.50510, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 9.0279 - acc: 0.3940 - val_loss: 9.5051 - val_acc: 0.3102
    Epoch 5/20
    6480/6680 [============================>.] - ETA: 0s - loss: 8.9258 - acc: 0.4103Epoch 00004: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 8.9221 - acc: 0.4111 - val_loss: 9.5120 - val_acc: 0.3257
    Epoch 6/20
    6520/6680 [============================>.] - ETA: 0s - loss: 8.8691 - acc: 0.4236Epoch 00005: val_loss improved from 9.50510 to 9.46979, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.8566 - acc: 0.4240 - val_loss: 9.4698 - val_acc: 0.3222
    Epoch 7/20
    6460/6680 [============================>.] - ETA: 0s - loss: 8.8035 - acc: 0.4313Epoch 00006: val_loss improved from 9.46979 to 9.34994, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.7785 - acc: 0.4325 - val_loss: 9.3499 - val_acc: 0.3425
    Epoch 8/20
    6620/6680 [============================>.] - ETA: 0s - loss: 8.6941 - acc: 0.4444Epoch 00007: val_loss improved from 9.34994 to 9.32077, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.6983 - acc: 0.4443 - val_loss: 9.3208 - val_acc: 0.3473
    Epoch 9/20
    6560/6680 [============================>.] - ETA: 0s - loss: 8.6650 - acc: 0.4502Epoch 00008: val_loss improved from 9.32077 to 9.29291, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.6740 - acc: 0.4494 - val_loss: 9.2929 - val_acc: 0.3509
    Epoch 10/20
    6540/6680 [============================>.] - ETA: 0s - loss: 8.6437 - acc: 0.4491Epoch 00009: val_loss improved from 9.29291 to 9.17853, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.6371 - acc: 0.4494 - val_loss: 9.1785 - val_acc: 0.3485
    Epoch 11/20
    6440/6680 [===========================>..] - ETA: 0s - loss: 8.4688 - acc: 0.4585Epoch 00010: val_loss improved from 9.17853 to 9.15698, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.4612 - acc: 0.4588 - val_loss: 9.1570 - val_acc: 0.3521
    Epoch 12/20
    6500/6680 [============================>.] - ETA: 0s - loss: 8.3949 - acc: 0.4642Epoch 00011: val_loss improved from 9.15698 to 8.95872, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.3817 - acc: 0.4648 - val_loss: 8.9587 - val_acc: 0.3689
    Epoch 13/20
    6480/6680 [============================>.] - ETA: 0s - loss: 8.2210 - acc: 0.4728Epoch 00012: val_loss improved from 8.95872 to 8.89037, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.1921 - acc: 0.4751 - val_loss: 8.8904 - val_acc: 0.3629
    Epoch 14/20
    6460/6680 [============================>.] - ETA: 0s - loss: 8.0870 - acc: 0.4859Epoch 00013: val_loss improved from 8.89037 to 8.79078, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 8.0625 - acc: 0.4873 - val_loss: 8.7908 - val_acc: 0.3832
    Epoch 15/20
    6520/6680 [============================>.] - ETA: 0s - loss: 7.9784 - acc: 0.4928Epoch 00014: val_loss improved from 8.79078 to 8.75020, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.9737 - acc: 0.4931 - val_loss: 8.7502 - val_acc: 0.3784
    Epoch 16/20
    6440/6680 [===========================>..] - ETA: 0s - loss: 7.9276 - acc: 0.4972Epoch 00015: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 7.8847 - acc: 0.4999 - val_loss: 8.8491 - val_acc: 0.3737
    Epoch 17/20
    6420/6680 [===========================>..] - ETA: 0s - loss: 7.8378 - acc: 0.5062Epoch 00016: val_loss improved from 8.75020 to 8.70251, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.8637 - acc: 0.5046 - val_loss: 8.7025 - val_acc: 0.3940
    Epoch 18/20
    6480/6680 [============================>.] - ETA: 0s - loss: 7.8304 - acc: 0.5097Epoch 00017: val_loss improved from 8.70251 to 8.69923, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.8507 - acc: 0.5082 - val_loss: 8.6992 - val_acc: 0.3928
    Epoch 19/20
    6620/6680 [============================>.] - ETA: 0s - loss: 7.7190 - acc: 0.5154Epoch 00018: val_loss improved from 8.69923 to 8.50322, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.7245 - acc: 0.5151 - val_loss: 8.5032 - val_acc: 0.4108
    Epoch 20/20
    6440/6680 [===========================>..] - ETA: 0s - loss: 7.6501 - acc: 0.5177Epoch 00019: val_loss improved from 8.50322 to 8.46931, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 1s - loss: 7.6331 - acc: 0.5189 - val_loss: 8.4693 - val_acc: 0.4096
    




    <keras.callbacks.History at 0x7fa1a004d5f8>



### Load the Model with the Best Validation Loss


```python
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

### Test the Model

Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.


```python
# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 40.9091%
    

### Predict Dog Breed with the Model


```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step5'></a>
## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features.  In this section, you must use the bottleneck features from a different pre-trained model.  To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras:
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

The files are encoded as such:

    Dog{network}Data.npz
    
where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  Pick one of the above architectures, download the corresponding bottleneck features, and store the downloaded file in the `bottleneck_features/` folder in the repository.

### (IMPLEMENTATION) Obtain Bottleneck Features

In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:

    bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']


```python
### TODO: Obtain bottleneck features from another pre-trained CNN.

# Using VGG-19 bottleneck features
bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
train_VGG19 = bottleneck_features['train']
valid_VGG19 = bottleneck_features['valid']
test_VGG19 = bottleneck_features['test']
```

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:
    
        <your model's name>.summary()
   
__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ 



__My anser:__

Firstly, I finished the step 4 above.

Secondly, I finished reading papers in [Alexnet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) and [ResNet](https://arxiv.org/pdf/1512.03385v1.pdf).

VGG16 architecture has done a fair job with 20 epochs training. Therefore, I am thinking with VGG19 architecture should be doing a better job with the same amout of training epochs since it has 3 more layers than the VGG16 has.

Finally, I decided to use VGG19 architecture. 
___
__Outline steps:__

Started from VGG19 architecture, I added "Global Average Pooling" layer to diminish the paramters to process.

Then it will go into a dense layer with RELU activation function.

To avoid overfitting, I put a drop-out layer with 0.5 (a good rule of thumb).

In the end, use a softmax layer to predict the result.
___


```python
### TODO: Define your architecture.

# Based on the previous architecture, I removed several layers since we are using VGG19

model_VGG19 = Sequential()
# Add a Global Average Pooling layer
model_VGG19.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
# Add a fully connected layer with RELU function
model_VGG19.add(Dense(500, activation='relu'))
# Add a dropout layer with 0.5
model_VGG19.add(Dropout(0.5))
# Return probabilities
model_VGG19.add(Dense(133, activation='softmax'))

model_VGG19.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_2 ( (None, 512)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 500)               256500    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 500)               0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 133)               66633     
    =================================================================
    Total params: 323,133.0
    Trainable params: 323,133.0
    Non-trainable params: 0.0
    _________________________________________________________________
    

### (IMPLEMENTATION) Compile the Model


```python
### TODO: Compile the model.

model_VGG19.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.  

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement. 


```python
### TODO: Train the model.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG19.hdf5', 
                               verbose=1, save_best_only=True)

model_VGG19.fit(train_VGG19, train_targets, 
          validation_data=(valid_VGG19, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/20
    6460/6680 [============================>.] - ETA: 0s - loss: 7.7548 - acc: 0.1537Epoch 00000: val_loss improved from inf to 2.11539, saving model to saved_models/weights.best.VGG19.hdf5
    6680/6680 [==============================] - 2s - loss: 7.5938 - acc: 0.1611 - val_loss: 2.1154 - val_acc: 0.4599
    Epoch 2/20
    6520/6680 [============================>.] - ETA: 0s - loss: 2.2699 - acc: 0.4710Epoch 00001: val_loss improved from 2.11539 to 1.18651, saving model to saved_models/weights.best.VGG19.hdf5
    6680/6680 [==============================] - 1s - loss: 2.2535 - acc: 0.4750 - val_loss: 1.1865 - val_acc: 0.6743
    Epoch 3/20
    6640/6680 [============================>.] - ETA: 0s - loss: 1.6748 - acc: 0.5968Epoch 00002: val_loss improved from 1.18651 to 1.10158, saving model to saved_models/weights.best.VGG19.hdf5
    6680/6680 [==============================] - 1s - loss: 1.6749 - acc: 0.5966 - val_loss: 1.1016 - val_acc: 0.6946
    Epoch 4/20
    6580/6680 [============================>.] - ETA: 0s - loss: 1.3982 - acc: 0.6549Epoch 00003: val_loss improved from 1.10158 to 0.97286, saving model to saved_models/weights.best.VGG19.hdf5
    6680/6680 [==============================] - 1s - loss: 1.4015 - acc: 0.6545 - val_loss: 0.9729 - val_acc: 0.7389
    Epoch 5/20
    6560/6680 [============================>.] - ETA: 0s - loss: 1.2828 - acc: 0.6910Epoch 00004: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 1.2831 - acc: 0.6909 - val_loss: 1.0987 - val_acc: 0.7329
    Epoch 6/20
    6500/6680 [============================>.] - ETA: 0s - loss: 1.1408 - acc: 0.7265Epoch 00005: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 1.1394 - acc: 0.7271 - val_loss: 1.0029 - val_acc: 0.7569
    Epoch 7/20
    6560/6680 [============================>.] - ETA: 0s - loss: 1.0989 - acc: 0.7393Epoch 00006: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 1.0988 - acc: 0.7397 - val_loss: 1.0482 - val_acc: 0.7449
    Epoch 8/20
    6500/6680 [============================>.] - ETA: 0s - loss: 1.0586 - acc: 0.7551Epoch 00007: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 1.0597 - acc: 0.7545 - val_loss: 1.0009 - val_acc: 0.7689
    Epoch 9/20
    6660/6680 [============================>.] - ETA: 0s - loss: 0.9776 - acc: 0.7724Epoch 00008: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 0.9794 - acc: 0.7720 - val_loss: 1.0761 - val_acc: 0.7689
    Epoch 10/20
    6500/6680 [============================>.] - ETA: 0s - loss: 0.9176 - acc: 0.7883Epoch 00009: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 0.9152 - acc: 0.7894 - val_loss: 1.1465 - val_acc: 0.7521
    Epoch 11/20
    6500/6680 [============================>.] - ETA: 0s - loss: 0.9611 - acc: 0.7882Epoch 00010: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 0.9624 - acc: 0.7883 - val_loss: 1.1641 - val_acc: 0.7641
    Epoch 12/20
    6480/6680 [============================>.] - ETA: 0s - loss: 0.8516 - acc: 0.8148Epoch 00011: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 0.8512 - acc: 0.8148 - val_loss: 1.3969 - val_acc: 0.7677
    Epoch 13/20
    6500/6680 [============================>.] - ETA: 0s - loss: 0.9012 - acc: 0.8069Epoch 00012: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 0.8977 - acc: 0.8078 - val_loss: 1.2852 - val_acc: 0.7617
    Epoch 14/20
    6500/6680 [============================>.] - ETA: 0s - loss: 0.8295 - acc: 0.8180Epoch 00013: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 0.8291 - acc: 0.8172 - val_loss: 1.2368 - val_acc: 0.7545
    Epoch 15/20
    6560/6680 [============================>.] - ETA: 0s - loss: 0.8400 - acc: 0.8221Epoch 00014: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 0.8408 - acc: 0.8210 - val_loss: 1.2350 - val_acc: 0.7653
    Epoch 16/20
    6540/6680 [============================>.] - ETA: 0s - loss: 0.7636 - acc: 0.8329Epoch 00015: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 0.7687 - acc: 0.8328 - val_loss: 1.4612 - val_acc: 0.7581
    Epoch 17/20
    6580/6680 [============================>.] - ETA: 0s - loss: 0.7694 - acc: 0.8386Epoch 00016: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 0.7644 - acc: 0.8394 - val_loss: 1.5728 - val_acc: 0.7653
    Epoch 18/20
    6540/6680 [============================>.] - ETA: 0s - loss: 0.7935 - acc: 0.8391Epoch 00017: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 0.7949 - acc: 0.8394 - val_loss: 1.6147 - val_acc: 0.7605
    Epoch 19/20
    6540/6680 [============================>.] - ETA: 0s - loss: 0.7998 - acc: 0.8454Epoch 00018: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 0.8013 - acc: 0.8451 - val_loss: 1.5257 - val_acc: 0.7689
    Epoch 20/20
    6520/6680 [============================>.] - ETA: 0s - loss: 0.7465 - acc: 0.8575Epoch 00019: val_loss did not improve
    6680/6680 [==============================] - 1s - loss: 0.7620 - acc: 0.8557 - val_loss: 1.5518 - val_acc: 0.7617
    




    <keras.callbacks.History at 0x7fa192ddb630>



### (IMPLEMENTATION) Load the Model with the Best Validation Loss


```python
### TODO: Load the model weights with the best validation loss.

model_VGG19.load_weights('saved_models/weights.best.VGG19.hdf5')
```

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.


```python
### TODO: Calculate classification accuracy on the test dataset.

# get index of predicted dog breed for each image in test set
VGG19_predictions = [np.argmax(model_VGG19.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG19]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG19_predictions)==np.argmax(test_targets, axis=1))/len(VGG19_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 71.8900%
    

### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan_hound`, etc) that is predicted by your model.  

Similar to the analogous function in Step 5, your function should have three steps:
1. Extract the bottleneck features corresponding to the chosen CNN model.
2. Supply the bottleneck features as input to the model to return the predicted vector.  Note that the argmax of this prediction vector gives the index of the predicted dog breed.
3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed.

The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.  To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function

    extract_{network}
    
where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.


```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

def VGG19_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model_VGG19.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
    
```

---
<a id='step6'></a>
## Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 5 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

import random 
from random import shuffle

def image_path_vector_shuffle(img_path_vector):
    shuffle(img_path_vector)

def image_plotting(img_path):
    # Display the image
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    plt.show()

def dog_breed_scanner(img_path):
    # Get the prediction from the model
    dog_breed = VGG19_predict_breed(img_path)
    
    # Making decision based on the detector
    if dog_detector(img_path):
        print("Hello, " + str(dog_breed) + "!")
        image_plotting(img_path)
        print("Need a walk?\n")
    elif face_detector(img_path):
        print("Hello, human!")
        image_plotting(img_path)
        print("You look like a " + str(dog_breed) + "\n")
    else:
        print("Sorry, I could not recognize you.")
        image_plotting(img_path)
        print("May I have your name?\n")
```


```python
# Shuffle dataset "test_files"
image_path_vector_shuffle(test_files)

# Predict first four images in the shuffled dataset "test_files"
for i in range(4):
    dog_breed_scanner(test_files[i])
```

    Hello, Chihuahua!
    


![png](./images/output_67_1.png)


    Need a walk?
    
    Hello, Japanese_chin!
    


![png](./images/output_67_3.png)


    Need a walk?
    
    Hello, Nova_scotia_duck_tolling_retriever!
    


![png](./images/output_67_5.png)


    Need a walk?
    
    Hello, Field_spaniel!
    


![png](./images/output_67_7.png)


    Need a walk?
    
    

---
<a id='step7'></a>
## Step 7: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that __you__ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ 

__My answer:__
The output is somehow not as good as I expected. It is doing a good job in predicting human or dog. However, it is not predicting well in dog breed when it  sees human face. One of the human should looks similar to Bull_terrier and there is another one looks similar Chinese_shar-pei. That's why I am putting those two dogs picture in my testing for visual comparision.

Here are few possible points of improvement for my algorithm:

1. Using different architecture like ResNet
2. Incresaing the training epochs to increase accuracy
3. Augumented the data set to have more data to train
4. Add more drop-out layer to avoid overfitting




```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

# Load and shuffle all the images from folder "My_Test_Images"
my_img_files = glob("My_Test_Images/*")
image_path_vector_shuffle(my_img_files)

# Predict first six images in the shuffled dataset "my_img_files"
for i in range(6):
    dog_breed_scanner(my_img_files[i])


```

    Hello, human!
    


![png](./images/output_70_1.png)


    You look like a Brittany
    
    Hello, human!
    


![png](./images/output_70_3.png)


    You look like a Dogue_de_bordeaux
    
    Hello, Bull_terrier!
    


![png](./images/output_70_5.png)


    Need a walk?
    
    Hello, human!
    


![png](./images/output_70_7.png)


    You look like a American_staffordshire_terrier
    
    Hello, human!
    


![png](./images/output_70_9.png)


    You look like a Dogue_de_bordeaux
    
    Hello, Chinese_shar-pei!
    


![png](./images/output_70_11.png)


    Need a walk?
    
    
