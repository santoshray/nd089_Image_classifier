import sys 
import os
import argparse  
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict
import re 
from pathlib import Path
from PIL import Image
import numpy as np

def load_pretrained_model(check_point_path):
    check_point = torch.load(check_point_path)
    arch = check_point['arch']
    model = models.vgg11(weights='DEFAULT')

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = check_point['classifier']
    model.to(device)
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = check_point['optimizer']
    model.class_to_idx = check_point['class_to_idx']
    model.load_state_dict(check_point['state_dict'])
    return model    


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)

    img_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image = img_transforms(img_pil).numpy()
    return image  


def predict(image_path, model, device , topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Done: Implement the code to predict the class from an image file
    #Covert label(idx) to class   to Class to label dict
    idx_to_class = model.class_to_idx
    class_to_label_dict = {}
    for label in idx_to_class.keys():
        cls = idx_to_class[label]
        class_to_label_dict[cls] = label    
    
    model.to(device)
    model.eval()
    img = process_image(image_path)
    img = torch.from_numpy(img).float().to(device)
    img = torch.unsqueeze(img, dim=0)
    
    output = model.forward(img)
    preds = torch.exp(output).topk(topk)
    ps = preds[0][0].cpu().data.numpy()
    classes = preds[1][0].cpu().data.numpy()
    topk_labels = [class_to_label_dict[i] for i in classes]
    return ps.tolist(), topk_labels



if __name__ == "__main__":

    #print(sys.argv)
    parser = argparse.ArgumentParser(
                    prog='predict.py',
                    description='python script to get the category of image  when an image is provided as input to trained classier ',
                    epilog = 'python predict.py flowers/test/2/image_05100.jpg check_point.pth --top_k 5 --category cat_to_name.json --gpu'
                    )
    
    parser.add_argument('path', default='',type=str,help='Provide the path of the image')           # positional argument
    parser.add_argument('checkpoint',type=str, default='check_point.pth', help='preTrained Model')      # option that takes a value
    parser.add_argument('--top_k',type=int,default = 5 , help =" return first k probable classes")  
    parser.add_argument('--category', default='cat_to_name.json', type=str, help='category to name json file')
    parser.add_argument('--gpu', default=False, action='store_true', help='GPU to be used for training?')

    args = parser.parse_args()
    #print("path={} , checkpoint={} , top_k ={} ,category ={}".format(args.path,args.checkpoint,args.top_k,args.category))
    
    img_path = str(args.path)
    checkpoint = str(args.checkpoint)
    top_k = int(args.top_k)
    category_file = str(args.category)
    gpu = bool(args.gpu)


    with open(category_file, 'r') as f:
        cat_to_name = json.load(f)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if gpu == True and device == "cuda:0":
        device ="cuda:0"
    else:
        device = "cpu"
    
    path = Path(img_path)
    if path.exists() == False:
        print("img_path = {} cannot be found".format(img_path))
        exit()

    model = load_pretrained_model(checkpoint)
    prob_topk,topk_category = predict(img_path,model,device,top_k)
    print("--------------: Prediction Result :--------------------")
    print("image_path = {}".format(img_path))
    print("prob_topk = {}".format([int(p*1000)/1000 for p in prob_topk]))
    print("topk_category = {}".format(topk_category))
    print("top_k_categories = {}".format([cat_to_name[category] for category in topk_category]))

    
