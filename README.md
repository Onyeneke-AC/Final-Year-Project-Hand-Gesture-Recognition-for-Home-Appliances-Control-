# Final-Year-Project-Hand-Gesture-Recognition-for-Home-Appliances-Control-
Use of a trained model to identify and classify 18 different hand gestures which was used for control of different home appliances

The model was trained using convolutional neural network (CNN) and then deployed on a raspberry pi 3B which used a pi camera to capture images placed in front of it. The captured images were then sent to the model via the image processing code. This returns a gesture index which corresponds to the captured image. The raspberry pi then uses this gesture index to control the appliances connected to it.

NB: The control was achieved via relay modules connected to the GPIO pins which either turn ON or OFF depending on the gesture index returned after image processing.
