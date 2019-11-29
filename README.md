# Deep-Learning_Image_Classifier_NN

## Project: Image Classifier project

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [PyTorch](https://pytorch.org/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)
Also there are 2 files - `train.py` and `predict.py` for which you will need to have software installed on some IDE, preferably Atom [Atom](https://atom.io/) 

Install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

### Code

3 files represent the code needed to build the Image classifier application
- `Image Classifier Project.ipynb`
- `train.py`
- `predict.py`

Note that the code included in `workspace_utils.py` is to be invoked as an external module to be able to keep session active in case the training of Neural NEtwork goes beyond 30 minutes

### GPU 
You can choose to run the program in GPU mode for which you need to have a remote server setup. For the sake of this project, i ran the program locally on UDacity's offered workspace which was GPU enabled

### Run

In a terminal or command window, navigate to the top-level project directory `Deep-Learning_Image_Classifier_NN/` (that contains this README) and run one of the following commands:

```bash
jupyter notebook Image Classifier project.ipynb
jupyter train.py
jupyter predict.py
```

This will open the iPython Notebook software and project file in your browser.
For the train.py and predict.py files, if you have Atom or any other IDE installed in your system, you can run the commands stated above in Anaconda prompt and execute the code in the IDE

### About the project
In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this dataset - http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html of 102 flower categories

## The project is broken down into multiple steps:
- Load and preprocess the image dataset
- Apply a pre-trained Pytorch model
- Build a Neural Network architecture with input, hidden and output layers
- Train the image classifier on your dataset 
- Use the trained classifier to predict image content

## Command Line Application
Also as a part of train.py and predict.py files, a command line Application program is build to do the following:
- Package Imports	- All the necessary packages and modules are imported in the first cell of the notebook
- Training data augmentation	- torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping
- Data normalization	- The training, validation, and testing data is appropriately cropped and normalized
- Data loading	- The data for each set (train, validation, test) is loaded with torchvision's ImageFolder
- Data batching	- The data for each set is loaded with torchvision's DataLoader
- Pretrained Network	- A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen
- Feedforward Classifier - 	A new feedforward network is defined for use as a classifier using the features as input
- Training the network	- The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static
- Validation Loss and Accuracy - 	During training, the validation loss and accuracy are displayed
- Testing Accuracy - 	The network's accuracy is measured on the test data
- Saving the model - 	The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary

## References
https://pytorch.org/docs/stable/torchvision/models.html
https://pytorch.org/docs/stable/torchvision/models.html
https://discuss.pytorch.org/t/cant-convert-a-given-np-ndarray-to-a-tensor/21321
https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data
https://discuss.pytorch.org/t/runtimeerror-input-type-torch-cuda-floattensor-and-weight-type-torch-floattensor-should-be-the-same/21782
https://stackoverflow.com/questions/2356925/how-to-check-whether-string-might-be-type-cast-to-float-in-python
https://discuss.pytorch.org/t/convert-a-jpegimagefile-to-a-tensor-to-train-a-resnet/9101
