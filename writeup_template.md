#**Behavioral Cloning** 

## Writeup 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidianet]: ./examples/network_architecture.PNG "NVIDIA Network Architecture"
[val_loss_1]: ./examples/val_loss_1.PNG "Loss Function over Epochs"
[val_loss_2]: ./examples/val_loss_2.PNG "Loss Function over Epochs"
[val_loss_3]: ./examples/val_loss_3.PNG "Loss Function over Epochs"
[val_loss_4]: ./examples/val_loss_4.PNG "Loss Function over Epochs"
[val_loss_5]: ./examples/val_loss_5.PNG "Loss Function over Epochs"

---

### Model Architecture and Training Strategy

The model architecture which has inspired my architecture was the one mentioned in the course videos. The arcticle from NVIDIA
(https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) explains their approach, and was detailed enough to
build a similar system. 

![alt text][nvidianet]

#### Model Architecture

The model of Nvidia was clearly tailored to real world application and overfitted the simple image data frrom the simulator by a huge margin.
To solve this problem, I introduced several dropout layers, to ensure a more robust model.

The overall model at this point lookes like follows

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image   						| 
| Image Normalization  	| 												|
| Image Cropping		| 320x65x3 RGB image							|
| Convolution 5 x 5	    | 2 x 2	stride, outputs 158x60x24				|
| Convolution 5 x 5	    | 2 x 2	stride, outputs 73x28x36 				|
| Convolution 5 x 5	    | 2 x 2	stride, outputs 33x12x48 				|
| Convolution 3 x 3	    | outputs 31x10x64 								|
| Convolution 3 x 3	    | outputs 29x8x64								|
| RELU					|												|
| Dropout				| 50% Dropout Rate								|
| Fully connected		| 60 Hidden Units								|
| Dropout				| 50% Dropout Rate								|
| Fully connected		| 30 Hidden Units								|
| Dropout				| 50% Dropout Rate								|
| Fully connected		| 10 Hidden Units								|
| Output Layer			| 1 Regression Value							|

This model of the same amount of layers as the original Nvidia Model but has a slightly different layer size. It also includes 4 dropout layers
to ensure, that the model generalizes well and doesn't overfit the data.

The following graph shows the first trainings, which were done with this model:

![alt text][val_loss_1]

#### Model parameter tuning

The first shot of the model was more or less copied from the Nvidia Model. To tune the parameters a few test runs were made, as mentioned below.

#### Appropriate training data

The first training data consisted of a normal lap, some rescue trips from the side of the road to the center, and most inportantly 5 rides over the bridge.
As the bridge in the frist track is heavily underrepresented I started recording runs from different directions, to ensure that this road texture
was captured well enough.

To ensure that all training data is used to its fullest, Left and Right cameras were used with the according angle adjusted by 0.15 degrees. Also, the inverted
image with inverted steering angle was used to alleviate the fact, that the track has a heavy shift to the left.

### Model and Trainign Improvements

The first run showed a good accuracy but tended to drive on the left side of the road. Which resulted in an immediate crash to the boundary after a few meters.
Due to my first training run being a bit rough and really fast, I decided to re-record the whole training with a smoother approach.

I kept the strategy of including a bridge run multiple times. The texture is way too different from the road to train it with just 2 runs. Also, until I could complete
a whole test run on the first track I didn't want to touch the second track provided in the simulator. The addition of right and left lanes, as well as shadows seemed
to different from the first track to start training just yet.

#### Solution Design Approach

With the new training data, the model started out to be much worse than before. Although the training loss was really good, it showed much more overfitting than in the first run.
A possible countermeasure was to reduce the layer size or introduce mode dropout layers.

![alt text][val_loss_2]

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image   						| 
| Image Normalization  	| 												|
| Image Cropping		| 320x65x3 RGB image							|
| Convolution 5 x 5	    | 2 x 2	stride, outputs 158x60x24				|
| Convolution 5 x 5	    | 2 x 2	stride, outputs 73x28x36 				|
| Convolution 5 x 5	    | 2 x 2	stride, outputs 33x12x48 				|
| Convolution 3 x 3	    | outputs 31x10x64 								|
| Convolution 3 x 3	    | outputs 29x8x64								|
| RELU					|												|
| Dropout				| 65% Dropout Rate								|
| Fully connected		| 60 Hidden Units								|
| Dropout				| 65% Dropout Rate								|
| Fully connected		| 30 Hidden Units								|
| Output Layer			| 1 Regression Value							|

The new model was now much smaller and had a larger dropout rate, to greatly reduce the validation loss. A benefit of the smaller model was also, that it trained much faster, due to the
smaller parameter space. It still generalized well and improved the validation loss a lot.

![alt text][val_loss_3]

The performance on the track was acceptable. It stayed in the middle of the road for most of the time, and got through the first large left curve flawlessly.
The performance on the brige was also acceptable, although it stayed a bit on the left. This can be due to unperfect testing data, I have to admit, my driving
skills are not that perfect after all.
In the narrow curves the car tended to stay on the outskirts of the curve, which is a rather unpleasant result. A way to improve this, is to get more data in those
2 curves. Like the bridge, this behavior is rather underrepresented in the model.

#### Model Exploration

Although the model performance was ok, I wanted to explore the effects of adding RELU Activations to the model. RELU Activation introduce more nonlinearities and may
be able to cope better with the task at hand. The best way was to test it on the already introduced model, and test the performance.
Also the dropout rate was reduced to 50% to see if a less robust training can lead to a better system.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image   						| 
| Image Normalization  	| 												|
| Image Cropping		| 320x65x3 RGB image							|
| Convolution 5 x 5	    | 2 x 2	stride, outputs 158x60x24				|
| Convolution 5 x 5	    | 2 x 2	stride, outputs 73x28x36 				|
| Convolution 5 x 5	    | 2 x 2	stride, outputs 33x12x48 				|
| Convolution 3 x 3	    | outputs 31x10x64 								|
| Convolution 3 x 3	    | outputs 29x8x64								|
| RELU					|												|
| Dropout				| 40% Dropout Rate								|
| Fully connected		| 60 Hidden Units								|
| RELU					|												|
| Dropout				| 40% Dropout Rate								|
| Fully connected		| 30 Hidden Units								|
| RELU					|												|
| Output Layer			| 1 Regression Value							|

The Training and Validation loss, show a similar performance than before. Validation and Training loss are even lower, than without RELU Activation.

![alt text][val_loss_4]

The new model, with slightly lower loss showed a less robust driving behavior. It didn't stay in the center of the road and failed in the curves. This led
to my decision, to stay with the more robust model.

Personally I preferred this architecture, because it performed better numerically and may probably be improved with better training data. Issues with the
training data that still persist are:
	- Lack of Bridge Data
	- Lack of Curve Data
	- Lack of Data with odd borders (Dirt Border at one part of the road)
	
In order to overcome those issues, I made another training run, recording just data from those parts of the track in both directions. Also in order to improve
the run over the bridge, I recorded multiple "saves", e.g. runs from the bridge border to the center.


### Final Model Architecture

The final model architecture looks as follows. The last improvements were made by gathering more data at critical points of the track.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image   						| 
| Image Normalization  	| 												|
| Image Cropping		| 320x65x3 RGB image							|
| Convolution 5 x 5	    | 2 x 2	stride, outputs 158x60x24				|
| Convolution 5 x 5	    | 2 x 2	stride, outputs 73x28x36 				|
| Convolution 5 x 5	    | 2 x 2	stride, outputs 33x12x48 				|
| Convolution 3 x 3	    | outputs 31x10x64 								|
| Convolution 3 x 3	    | outputs 29x8x64								|
| RELU					|												|
| Dropout				| 65% Dropout Rate								|
| Fully connected		| 60 Hidden Units								|
| Dropout				| 65% Dropout Rate								|
| Fully connected		| 30 Hidden Units								|
| Output Layer			| 1 Regression Value							|

The loss function show a quite robust behavior numerically, and behave equally well on the test track. The improvements which were made to
the training data also showed improvements in the autonomous behavior.
If I manouvered the car to the side of the bridge, it was now able to navigate back to the center, a behavior which couldn't be observed before.

The larger training dataset 24000 images instead of 16000 images, also meant, that overfitting would be less of an issue, and that training for more
than 10 epochs may improve the validation loss. To test this assumption the model was trained for 20 epochs instead of 10.

![alt text][val_loss_5]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
