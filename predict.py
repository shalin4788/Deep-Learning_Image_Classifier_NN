
import json
import numpy as np
import torch
import os, random
from PIL import Image
from torch.autograd import Variable

def load_json():
    print('Loading json file:\n')
    #filename = input('Enter the name of json file you want to read:').lower()
    with open('ImageClassifier/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    # Print the class values to category names
    print(cat_to_name)
    print("\n Length:", len(cat_to_name))
    print('-'*40)


def model_checkpoint():
    # Load checkpoint and print model details
    checkpoint = torch.load('checkpoint.pth')
    model = checkpoint['model']

    print('-'*40)
    print(model)
    return model

def gpu_mode(model):
    # Start gpu mode based on user input
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

    print('-'*40)
    return model

def top5_predict(model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    # Read the processed image and convert into tensor
    # turn off dropout
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
        print('Reading a hardcoded image from flowers path\n')
    else:
        model.cpu()
        print('Reading a hardcoded image from flowers path\n')
    # turn off dropout
    model.eval()

    # Randomly read an image from one of the test folders
    img = random.choice(os.listdir('ImageClassifier/flowers/test/7/'))
    img_path = 'ImageClassifier/flowers/test/7/' + img
    #Reize and crop the image picked up, normalize it
    im = Image.open(img_path)
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std
    im = im.transpose(2,0,1)

    # Convert array to tensor
    image = torch.from_numpy(np.array([im])).float()
    #print(image)
    image = Variable(image)
    if cuda == True:
        image = image.cuda()

    output = model.forward(image)
    ps = torch.exp(output).data

    # Get top k classes with highest probability
    prob = torch.topk(ps, topk)[0].tolist()[0]
    index = torch.topk(ps, topk)[1].tolist()[0]

    # Get indices related to top 5 classes
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # Fetch the label corresponding to indices of the top 5 classes
    label = []
    for i in range(5):
        label.append(ind[index[i]])

    return prob, label

def top_5_print(prob, label, model):
    prob, classes = top5_predict(model)
    with open('ImageClassifier/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    class_array = [cat_to_name[x] for x in classes]
    print('Top 5 class predictions related to the image are {}:'.format(class_array))
    #print('Indices corresponding to the top 5 classes are {}:'.format(classes))
    print('Probabilities of the top 5 image predicitons are {}:'.format(prob))

    print('-'*40)
    return prob, class_array


def max_prob_class(prob, class_array):
    max_index = np.argmax(prob)
    max_probability = prob[max_index]
    label = class_array[max_index]

    print('-'*40)
    print('Name of the top image prediction is {}:'.format(label))
    print('Probability of the top image prediction is {}:'.format(max_probability))

def main():
    while True:
        load_json()
        model = model_checkpoint()
        model = gpu_mode(model)
        prob, label = top5_predict(model, topk = 5)
        prob, class_array = top_5_print(prob, label, model)
        max_prob_class(prob, class_array)

        restart = input('\nWould you like to restart? Enter yes or no.\n')
        if restart.lower() != 'yes':
            break

if __name__ == "__main__":
    main()
