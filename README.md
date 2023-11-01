# Final-Year-Project-Hand-Gesture-Recognition-for-Home-Appliances-Control-
Use of a trained model to identify and classify 18 different hand gestures which was used for control of different home appliances.


The model was trained using a convolutional neural network (CNN) as seen in the train_model. The hand gestures used in training the CNN model was gotten from the Kaggle hand gesture dataset. The trained model was then deployed on a raspberry pi 3B which used a pi camera to capture images placed in front of it. The captured images were then sent to the model via the project_detect file. This returns a gesture index which corresponds to the captured image. The raspberry pi then uses this gesture index to control the appliances connected to it as seen in project_control.

The model was tested using test_model and it achieved an accuracy of 99.48%.

The entire project was further documented in Project Report and also presented using Onyeneke Anthony Proposal. 

NB: The control was achieved via relay modules connected to the GPIO pins which either turn ON or OFF depending on the gesture index returned after image processing.
