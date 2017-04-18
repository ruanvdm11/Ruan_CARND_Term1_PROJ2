# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ2/master/Visualization.PNG "Visualization"
[image2]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ2/master/Preprocessing_Normalisation.PNG "Normalisation"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ2/master/i1.png "Traffic Sign 1"
[image5]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ2/master/i23.jpg "Traffic Sign 2"
[image6]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ2/master/i3.jpg "Traffic Sign 3"
[image7]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ2/master/i4.jpg "Traffic Sign 4"
[image8]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ2/master/i5.jpg "Traffic Sign 5"
[image9]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ2/master/i6.jpg "Traffic Sign 6"
[image10]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ2/master/Network_Architecture2.png "Network Architecture"
[image11]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ2/master/Training_Process.PNG "Training Process"

## Rubric Points
##### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
Here is a link to my [project code](https://github.com/ruanvdm11/Ruan_CARND_Term1_PROJ2/blob/master/Traffic_Sign_Classifier.ipynb)

### 1. Data Set Summary & Exploration

#### 1.1. This section provides a summary of the data that was used for the creation of the neural network.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

| Summary Statistics of the Data||
|:---------------------										|:------------	|
| The size of training set is:								|34799			| 
| The size of test set is:									|12630			| 
| The shape of a traffic sign image is:						|32x32x3		|
| The number of unique classes/labels in the data set is:	|42				|
||||

#### 1.2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### 2. Design and Test a Model Architecture

#### 2.1. Image Preprocessing

The code for this step is contained in the fifth code cell of the IPython notebook.

I initially decided to run the neural network in order to determine, for myself, the benchmark in a case where no preprocessing was done. The result was unexpected because the model trained quite quickly and was obtaining good validation results (greater than 0.9). However, when newly imported images were tested the robustness of the network came into question. Therefore, I deemed it necessary to add a preprocessing section.

As a first step, I decided to normalize the images with the OpenCV function 'Normalize - MINMAX'. This function takes the rgb values from each pixel and sums them. Then each rgb value is divided by this sum. The reason for normalizing an image is to minimize the effect that light has in image processing.

Here is an example of a traffic sign image before and after normalization.

![alt text][image2]

I deemed it not necessary to greyscale the images as well. The reason for my decision is that the time necessary to train the model did not decrease significantly when the images were converted to grayscale. Also, with colour parameters also part of the training I believe, especially after normalization, that a more robust network is obtained as opposed to a network where grayscaling is implemented.

#### 2.2. Model Architecture

The code for the network architecture can be found in the seventh code cell of the IPython notebook.

For the Neural Network Structure I implemented the LeNet architecture adapted so that colour images are used. The network contains five layers and the input and output sizes of each layer is as follows:

| Layer Number		| Input Size	| Output Size	| Description												|
|:-----------		| :------------	| :---------	| :-------													|
| **Layer 1**		| **32x32x3**	| **28x28x6**	| **Convolutional Layer Stride=[1x1], Padding=['VALID']**	|
| Activation		| 				| 				| RELU														|
| Pooling Layer		| 28x28x6		| 14x14x6		| Pooling Layer Stride=[2x2], Padding=['VALID']				|
| **Layer 2**		| **14x14x6**	| **10x10x16**	| **Convolutional Layer Stride=[1x1], Padding=['VALID]**	|
| Activation		| 				| 				| RELU														|
| Pooling Layer		| 10x10x16		| 5x5x16		| Pooling Layer Stride=[2x2], Padding=['VALID']				|
| Flattening Layer	| 5x5x16		| 400			| Flattening												|
| **Layer 3**		| **400**		| **120**		| **Fully Connected Layer**									|
| Activation		| 				| 				| RELU														|
| **Layer 4**		| **120**		| **84**		| **Fully Connected Layer**									|
| Activation		| 				| 				| RELU														|
| **Layer 5**		| **84**		| **42**		| **Fully Connected Layer**									|
||||||||

The architecture of the model can be visualized by the following figure:

![alt text][image10]

The convolution of the network is clearly visible. The use of a convolutional network allows for the depth ("rgb") to be _squeezed_ to a shape that can be directly used for classification. Thus, the reason that the output of layer 5 is 42 is because there are 42 different classified road signs, or classes.

#### 2.3. Training, Validation, and Testing Data

The code for the network architecture can be found in the seventh code cell of the IPython notebook.

From the training data that was provided it was necessary to extract some of the data in order to validate the model. This was done using the *sklearn* library, and the "train_test_split" function.

According to the study material provided in one of the lessons it was stated that using 20% of the training data as validation data is the _rule of thumb_ therefore, this is the value that was used. Thus, the **validation set consisted of 6959 images** and the **training data reduced to 27840 images**.

There were no changes made to the **test set therefore, it still consists of 12630 images.** 


#### 2.4. Training and Validation.

The code for training the model is located in the ninth cell of the ipython notebook. 

To train the model, I used a methodology illustrated by the following figure. Thus, the training operation is initiated with a batch of X data and Y data (Which is simply images and their classification). A softmax is completed from the LeNet Neural Network Architecture in order to obtain _logits_. A loss operation is completed where the tf.reduceMean function is used on the softmax results. This is then passed through the Adam Optimizer, which uses a Stochastic Optimisation method.

The results used in this process is then used for validation in order for the accuracy to be determined during training.

![alt text][image11]

The epochs used for training was found by utilizing an iterative process. The model was trained a few times using varied Epoch and Batch Size values. The goal was to spend a minimum time training for a acceptable accuracy. Also, using more epochs than necessary could lead to overtraining of the model therefore, as soon as the training accuracy started to decrease thrice successively the epoch value was lowered. The learning rate was left at 0.001 because after iterations, this yielded the most acceptable time to train.

#### 2.5. Model Results and Applicability

The code for calculating the accuracy of the model is located in the tenth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 94.3%
* validation set accuracy of 85.6% 
* test set accuracy of 83.3%

The architecture that was chosen was the Lenet architecture used as a guide during the lessons. The reason for the choice of the architecture is that seeing as it was used for images in the lessons, and yieled satisfactory results it was deemed that it would be acceptable in this application.

As seen from the quoted model results (above) it is quite clear that the model is working well. However, the real test is when the model is confronted with new images. The behaviour of the model in these circumstances are shown in the following section and these results definitely show a successful model.
 

### 3. Test a Model on New Images

#### 3.1. Six New German Traffic Sign Images

Here are six German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

* The first image is quite straight forward "STOP" sign.
* The second image (Keep Right) has got some "noise" around it with colours that might throw it off.
* The third image (Priority Road) is rotated a bit with some writing across the image.
* The fourth image (Speed Limit 50) is another straight forward image. But the speed limit images look very similar which makes them more difficilt to classify.
* The fifth image (No Entry) has got some graffiti on the sign which might make it difficult to classify.
* The sixth and final image (Ahead only) is a bit different at the top of the arrow than usual "ahead only" signs. 

#### 3.2. Model Predictions and Relative Accuracy

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

Here are the results of the prediction:

| Image					| Prediction			 						| 
|:---------------------:| :----------------------------------------:	| 
| Stop Sign				| Stop Sign										| 
| Keep Right			| Keep Right									|
| Priority Road			| Priority Road									|
| 50 km/h				| 50 km/h					 					|
| No Entry				| Stop Sign										|
| Ahead Only			| Ahead Only									|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This compares favourably to the accuracy on the test set of 85.6%

#### 3.3. Certainty of Model Predictions Through Softmax Values

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is very sure that this is a Stop sign (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------	|:---------------------------------------------	| 
| 0.99         			| Stop sign   									| 
| 8.8e-05  				| No Entry 										|
| 1.6e-11				| Children Crossing								|
| 1.0e-14      			| Speed Limit 60km/h			 				|
| 4.1e-15			    | Dangerous Curve to the right					|


For the second image the model was immensely sure that the image was a Keep Right sign, which indeed it was. Here follows the top 5 Softmax probabilities:

| Probability         	|     Prediction	        					| 
|:---------------------	|:---------------------------------------------	| 
| 1.0         			| Keep Right   									| 
| 4.9e-09  				| Turn left ahead 								|
| 2.1e-34				| Ahead only									|
| 1.3e-38      			| End of no passing			 					|
| 1.3e-38			    | Yield											|


For the third image the model predicted a Priority Road sign which was the sign in the image. Here follows the Softmax probabilities:

| Probability         	|     Prediction	        					| 
|:---------------------	|:---------------------------------------------	| 
| 1.0         			| Priority road   								| 
| 8.6e-16  				| Double curve 									|
| 7.4e-16				| Slippery road									|
| 3.2e-16      			| No passing for vehicles over 3.5 metric tons	|
| 4.2e-17			    | Right-of-way at the next intersection			|


For the fourth image the model a 50km/h speed limit sign which was indeed the case. It is clear the the model is well trained w.r.t. speed limit signs because from inspection of the softmax probabilities it is clear that the difference is clear between the speed limit signe. Here are the corresponding probabilities:

| Probability         	|     Prediction	        					| 
|:---------------------	|:---------------------------------------------	| 
| 0.99        			| Speed limit (50km/h)   						| 
| 0.002  				| Speed limit (30km/h) 							|
| 0.0001				| Speed limit (80km/h)							|
| 2.0e-11      			| Speed limit (60km/h)							|
| 2.3e-16			    | Speed limit (100km/h)							|

For the fifth image the model predicted a Stop Sign and a No Entry Sign was contained in the image. The confusion by the model can be understood becasue the two traffic signs are similar. However, the model was too sure that it was a Stop Sign which is something that can be looked at when improving the model. Here are the top 5 softmax probabilities.

| Probability         	|     Prediction	        					| 
|:---------------------	|:---------------------------------------------	| 
| 0.99        			| Stop   										| 
| 0.001  				| Bicycles crossing 							|
| 0.001					| No entry										|
| 0.0002     			| Children crossing								|
| 6.6e-08			    | Bumpy road									|

For the sixth image the model predicted the correct traffic sign by classifying the image as an Ahead Only sign, which it was. Here are the softmax values:

| Probability         	|     Prediction	        					| 
|:---------------------	|:---------------------------------------------	| 
| 1.0        			| Ahead only   									| 
| 1.59159e-08  			| Turn left ahead 								|
| 2.52093e-11			| Priority road									|
| 2.51358e-13     		| Go straight or left							|
| 4.70888e-14		    | Roundabout mandatory							|

### 4. Conclusion

From the document and code submitted in this report it is clear that the created Neural Network, with the goal of classifying traffic signs, really did succeed in its task. There are however always improvements that can be made. In this case one of the improvements that can be to increase the robustness of the model is by also training the model with rotated images.