import sys 
import os
import argparse  

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict
import re 
from pathlib import Path



supported_models_set = set(['vgg11','vgg13','vgg16'])
def prepare_data(train_dir,valid_dir,test_dir):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
    test_transforms =  transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
    
    # TODO: Load the datasets with ImageFolder
    traindata = datasets.ImageFolder(train_dir, transform=train_transforms)

    testdata = datasets.ImageFolder(test_dir, transform=test_transforms)

    validdata = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testdata,batch_size=64,shuffle=True)
    validloader = torch.utils.data.DataLoader(validdata,batch_size=64,shuffle=True)

    data_dict = {}
    data_dict['trainloader'] = trainloader
    data_dict['testloader']  = testloader
    data_dict['validloader'] = validloader
    data_dict['traindata'] = traindata 
    data_dict['testdata'] = testdata
    data_dict['validata'] = validdata
    
    return data_dict 



def create_model(arch,device,learning_rate=0.01,hidden_layers=1024,drop_prob=0.1):
    
    if arch not in supported_models_set:
        print("Error : create_model : Supported arch :{} , returing None".format(list(supported_models_set)))
        return None
    
    print("Info:create_model: arch ={},device={},learning_rate={},hidden_layers={}".format(arch,device,learning_rate,hidden_layers))
    if arch == 'vgg11':
        model = models.vgg11('DEFAULT')
    elif arch =='vgg13':
        model = models.vgg13('DEFAULT')
    elif arch == 'vgg16':
        model = models.vgg16('DEFAULT')
    else:
        model = models.vgg11('DEFAULT')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
    model.to(device)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, hidden_layers)),
                            ('relu', nn.ReLU()),
                            ('drop_out',nn.Dropout(drop_prob)),
                            ('fc2', nn.Linear(hidden_layers, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    

    
    model.classifier =classifier
    criterion = nn.NLLLoss()  # Negative Log Likelihood Loss
   
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    model_dict = {}
    model_dict['model'] = model
    model_dict['arch']  = arch
    model_dict['criterion'] = criterion
    model_dict['class_to_idx'] = None
    model_dict['classifier'] = model.classifier
    model_dict['input_size'] = 25088
    model_dict['hidden_layer_size'] = hidden_layers
    model_dict['output_size'] = 102
    model_dict['learning_rate'] = learning_rate
    model_dict['optimizer'] = optimizer
    model_dict['drop_prob'] = drop_prob
    
    return model_dict 


def get_trained_model(model_dict,device,epochs=3,data_dict=None):
    
    print("Info:get_trained_model : device={},epochs={},data_dict={}".format(device,epochs,data_dict))
    
    if data_dict == None:
        print('Error:get_trained_model : There is no data_dict passed ')
        return None
    
    if model_dict == None :
        print('Error:get_trained_model : Model dict is None ')
        return None
        
  
    model = model_dict['model']
    trainloader = data_dict['trainloader']
    validloader = data_dict['validloader']
    testloader = data_dict['testloader']
    
    traindata = data_dict['traindata']
    validata  = data_dict['validata']
    testdata = data_dict['testdata']
    
    criterion = model_dict['criterion']
    optimizer = model_dict['optimizer']
    learning_rate = model_dict['learning_rate']
    
    print("training and  validation of model started at time = {}".format(time.ctime()))
    s_time = time.time_ns()
    
    #Start Training of the Model #
    #Move the Model to device 
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        print("epoch = {}".format(epoch+1))
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    
    
    print("training and  validation of model ended  at time = {}".format(time.ctime()))
    e_time = time.time_ns()
    print("total_time in secs = {}".format((e_time-s_time)/10**9))
    
    model.class_to_idx = traindata.class_to_idx
    model_dict['class_to_idx'] = model.class_to_idx
    return model_dict


def save_tranined_model(model_dict,path='checkpoint.pth'):
    if path == None :
        print("Error:save_tranined_model path is None")
    
    model_dict_to_save = {}
    model_dict_to_save['arch'] = model_dict['arch']
    model_dict_to_save['input_size'] = model_dict['input_size']
    model_dict_to_save['output_size'] = model_dict['output_size']
    model_dict_to_save['classifier'] = model_dict['classifier']
    model_dict_to_save['hidden_layer_size'] = model_dict['hidden_layer_size']
    model_dict_to_save['learning_rate'] = model_dict['learning_rate']
    model_dict_to_save['state_dict'] = model_dict['model'].state_dict()
    model_dict_to_save['optimizer'] = model_dict['optimizer'].state_dict()
    model_dict_to_save['class_to_idx'] = model_dict['class_to_idx']
    print("Info: save_trained_model to {}".format(path))
    torch.save(model_dict_to_save,path)
    return 



if __name__ == "__main__":

    print(sys.argv)
    parser = argparse.ArgumentParser(
                    prog='train.py',
                    description='python script to train a pretrained model to do Image Classification',
                    epilog = 'python train.py flowers --arch vgg11 --learning_rate 0.001 --drop_prob 0.1 --save_dir check_point.pth --hidden_units 512 --epochs 3 --gpu'
                    )
    
    parser.add_argument('data_dir', default='',type=str,help='Provide the path of the data dir')           # positional argument
    parser.add_argument('--arch',type=str, default='vgg11', help='models Architecture Type')      # option that takes a value
    parser.add_argument('--learning_rate',type=float,default = 0.001 , help ="Learning rate for the network ")  # on/off flag
    parser.add_argument('--drop_prob', default=0.1, type=float, help='dropout probability')
    parser.add_argument('--save_dir', default='', type=str, help="directory to save trained model")
    parser.add_argument('--hidden_units',type=int,default =1024, help="number of perceptrons in the hidden layer of the classifier")  # on/off flag
    parser.add_argument('--epochs', type=int, default=3, help="Number of times the traning loop in cariied out")  #number of epochs
    parser.add_argument('--gpu', default=False, action='store_true', help='GPU to be used for training?')

    args = parser.parse_args()
    #print(args.data_dir, args.arch, args.learning_rate,args.hidden_units,args.epochs)
    
    arch = str(args.arch)
    learning_rate = float(args.learning_rate)
    epochs = int(args.epochs) 
    hidden_layers = args.hidden_units
    drop_prob = args.drop_prob
    gpu = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_dir  = args.data_dir
    if data_dir == None :
        exit()
    path = Path(data_dir)
    if path.exists():
        train_dir = os.path.join(path ,'train')
        valid_dir = os.path.join(path, 'valid') 
        test_dir = os.path.join(path ,'test')
        if Path(train_dir).exists() == False or  Path(valid_dir).exists() == None or Path(test_dir).exists() == None:
            print("One of data_dir does not exist")
            exit()
    else:
        print("data_dir = {} cannot be found".format(data_dir))
        exit()
    
    model_dict = create_model(arch,device,learning_rate,hidden_layers,drop_prob)
    data_dict = prepare_data(train_dir,valid_dir,test_dir)
    model_dict = get_trained_model(model_dict,device,epochs,data_dict)
    save_tranined_model(model_dict,path)