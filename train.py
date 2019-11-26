# train.py successfully trains a new network on a dataset of images and saves the model to a checkpoint

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time

# Import workspace utils module to call active_session and keey_awake functions
from workspace_utils import active_session, keep_awake

model_name = {'vgg16':'vgg16',
              'squeezenet': 'squeezenet1_0',
              'densenet' :'densenet161'}

def load_data():
    data_dir = 'ImageClassifier/flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    test_loader  = torch.utils.data.DataLoader(test_data, batch_size=32)
    dataloaders = [train_loader, valid_loader, test_loader]
    print('-'*40)
    return train_data, valid_data, test_data, train_loader, valid_loader, test_loader, dataloaders

def choose_model(model_name):
    # Choose a new pretrained model
    model_name = input("Hello! Please choose what torchvision model you want to apply for pre-training the dataset. You can type vgg16 OR densenet OR squeezenet:").lower()
    try:
        if model_name == 'vgg16':
            model = models.vgg16(pretrained=True)

            for param in model.parameters():
                param.requires_grad = False

        elif model_name == 'squeezenet':
            model = models.squeezenet1_0(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False

        elif model_name == 'densenet':
            model = models.vgg16(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False

        else:
            print("Sorry {} is not a valid model name. Let's move on.".format(model_name))

    except ValueError as e:
        print("Exception occurred: {}".format(e))

    print('-'*40)
    return model

def define_architecture(model):
    # Define architecture of the model where hidden_layers can be set by user
    hidden_layers = int(input("Please build classifier by entering the number of hidden layers:"))
    learning_rate = float(input("Please set the learning rate as a decimal value:"))
    try:
        if (type(hidden_layers) == int and type(learning_rate) == float):
            classifier = nn.Sequential(OrderedDict([
                                      ('fc1', nn.Linear(25088, hidden_layers)),
                                      ('drop', nn.Dropout(p=0.5)),
                                      ('relu', nn.ReLU()),
                                      ('fc2', nn.Linear(hidden_layers, 102)),
                                      ('output', nn.LogSoftmax(dim=1))
                                      ]))
            model.classifier = classifier
            model
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
        else:
            if hidden_layers is not numeric:
                print("Sorry, {} is not a valid value. Please type an integer value".format(hidden_layers))
            elif learning_rate is not float:
                print("Sorry, {} is not a valid value. Please type an integer OR float value".format(learning_rate))
                #break
    except ValueError as e:
        print("Exception occurred: {}".format(e))

    print('-'*40)
    return model, classifier, criterion, optimizer

def gpu_mode(model):
    gpu_input = input("Type Y if you want to train model in GPU mode, else type N:").lower()
    if gpu_input == 'y':
        cuda = torch.cuda.is_available()
        if cuda:
            model = model.cuda()
        else:
            model = model.cpu()
    elif gpu_input == 'n':
        model = model.cpu()
    else:
        print("Invalid input. Please enter either Y or N")
        #continue

    return model

def training_loss(model, optimizer, criterion, dataloaders):
    # Train network and find training loss
    epochs_t = int(input("Enter the number of epochs the training set should iterate over:"))
    try:
        if type(epochs_t) == int:
            steps = 0
            running_loss = 0
            accuracy = 0
            epochs = epochs_t
            cuda = torch.cuda.is_available()
            if cuda:
                model.cuda()
            else:
                model.cpu()
            with active_session():
                for e in range(epochs):
                    train_mode = 0
                    valid_mode = 1
                    for mode in [train_mode, valid_mode]:
                        if mode == train_mode:
                            model.train()
                        else:
                            model.eval()

                        pass_count = 0

                        for data in dataloaders[mode]:
                            pass_count += 1
                            inputs, labels = data
                            if cuda == True:
                                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                            else:
                                inputs, labels = Variable(inputs), Variable(labels)

                            optimizer.zero_grad()
                            # Forward
                            output = model.forward(inputs)
                            loss = criterion(output, labels)
                            # Backward
                            if mode == train_mode:
                                loss.backward()
                                optimizer.step()

                            running_loss += loss.item()
                            ps = torch.exp(output).data
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        if mode == train_mode:
                            print("\nEpoch: {}/{} ".format(e+1, epochs),
                                  "\nTraining Loss: {:.4f}  ".format(running_loss/pass_count))
                        else:
                            print("Validation Loss: {:.4f}  ".format(running_loss/pass_count),
                                  "Accuracy: {:.4f}".format(accuracy))

                        running_loss = 0
        else:
            print("Sorry, {} is not a valid value. Please type an integer value".format(epochs))
            #break
    except ValueError as e:
        print("Exception occurred: {}".format(e))

    print('-'*40)
    return model, epochs


def testing_accuracy(model, optimizer, criterion, test_loader):
    # Find testing accuracy
    try:
        steps = 0
        accuracy = 0
        cuda = torch.cuda.is_available()
        if cuda:
            model.cuda()
        else:
            model.cpu()
        with active_session():
            for inputs, labels in iter(test_loader):
                steps += 1
                # Move input and label tensors to the default device
                if cuda == True:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                output = model.forward(inputs)
                ps = torch.exp(output).data
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print("\Testing Accuracy: {:.3f}".format(accuracy/steps))
    except ValueError as e:
        print("Exception occurred: {}".format(e))

    print('-'*40)
    return model

def save_model_Checkpoint(train_data, valid_data, test_data, model, optimizer, classifier,epochs):
    image_datasets = [train_data, valid_data, test_data]
    model.class_to_idx = image_datasets[0].class_to_idx
    checkpoint = {'model': model,
                  'classifier': classifier,
                  'optimizer' : optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'epochs' : epochs,
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, 'checkpoint.pth')
    print('-'*40)

def main():
    while True:
        train_data, valid_data, test_data, train_loader, valid_loader, test_loader, dataloaders = load_data()
        model = choose_model(model_name)
        model, classifier, criterion, optimizer = define_architecture(model)
        model = gpu_mode(model)
        model, epochs = training_loss(model, optimizer, criterion, dataloaders)
        #model = validation_loss(model, optimizer, criterion, valid_loader)
        model = testing_accuracy(model, optimizer, criterion, test_loader)
        save_model_Checkpoint(train_data, valid_data, test_data, model, optimizer, classifier, epochs)

        restart = input('\nWould you like to restart? Enter yes or no.\n')
        if restart.lower() != 'yes':
            break

if __name__ == "__main__":
    main()
