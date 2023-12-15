# Medical_Waste_segregation
 The Medical Waste Segregation using Machine Learning project is a significant endeavor that addresses the pressing issue of medical waste management. By integrating machine learning technology, we aim to enhance the accuracy and efficiency of waste segregation, contributing to a safer and more sustainable healthcare environment. This code is part of a project that uses TensorFlow and PyTorch for image classification. The project appears to be focused on classifying medical waste images. Here's a breakdown of the code:

Setup and Data Preparation

Installation of Dependencies: Installs TensorFlow, TensorFlow GPU, OpenCV, and Matplotlib using pip.
TensorFlow Setup: Imports TensorFlow, prints its version, and checks for GPU availability.
List Installed Packages: Lists all installed Python packages in the environment.
PyTorch and Other Libraries: Imports PyTorch, its neural network module, optimizer, data utilities, torchvision for image transformations, PIL for image processing, Matplotlib for plotting, scikit-learn for metrics, seaborn for visualization, OpenCV, and imghdr for image file format checking.
GPU Configuration in TensorFlow: Configures TensorFlow to use GPU efficiently by setting memory growth.
Mount Google Drive: Mounts Google Drive to access the dataset stored there.
Dataset Exploration: Explores the dataset by listing the number of images in each class in the training, testing, and validation directories.
Custom Dataset Class 8. Custom Dataset Class: Defines a custom dataset class for PyTorch that loads images and their labels from a directory.

Data Augmentation and Loading 9. Data Augmentation: Defines transformations for data augmentation and normalization for training data. 10. Data Loaders: Creates data loaders for training, validation, and testing datasets using the custom dataset class and transformations.

Model Preparation 11. Pre-trained Model: Loads a pre-trained ResNet50 model and freezes its layers for feature extraction. 12. Model Modification: Modifies the final layer of the model to match the number of classes in the dataset. 13. Optimizer and Loss Function: Sets up the Adam optimizer and cross-entropy loss function for training.

Training and Validation 14. Training Loop: Trains the model for a specified number of epochs, calculating training and validation loss and accuracy. 15. Accuracy and Loss Plotting: Plots training and validation accuracy over epochs.

Testing and Evaluation 16. Test Dataset and Loader: Prepares the test dataset and loader. 17. Model Evaluation: Evaluates the model on the test dataset, computes the confusion matrix, and visualizes it. 18. Accuracy Plotting: Plots training and validation accuracy over epochs.

Prediction and Visualization 19. Prediction Function: Defines a function for making predictions with the model. 20. Class Names: Specifies the class names for the dataset. 21. Image Denormalization: Defines a function to denormalize images for visualization. 22. Image Visualization: Visualizes a batch of images with their predicted and actual labels.
