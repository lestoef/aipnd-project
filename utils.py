import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from PIL import Image
import json

def process_image(image, show=False):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image) as im:
        width, height = im.size
        if min(width, height) < 256:
            raise ValueError
        else:
            if width < height:
                height = int(256 / width * height)
                width = 256
            else:
                width = int(256 / height * width)
                height = 256
        left = (width - 224)/2
        upper = (height - 224)/2
        right = (width + 224)/2
        lower = (height + 224)/2

        im = im.resize((width, height)) \
               .crop((left, upper, right, lower))
        if show:
            im.show()

        np_image = np.array(im)
        # normalize to values between 0 an 1
        np_image = np_image * 1./np_image.max() 
        
        # use the mean and standard deviation given to normalize as required by the pretrained model
        std = np.array([0.229, 0.224, 0.225])
        mean = np.array([0.485, 0.456, 0.406])
        np_image = (np_image - mean) / std
        
        # transpose, so the color channel is in the right position
        np_image = np_image.transpose((2, 0, 1))
        
    return np_image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    np_image = process_image(image_path)
    np_image = np_image.astype('float32')
    image_tensor = torch.from_numpy(np_image)
    model.eval()
    with torch.no_grad():
        predictions = model.forward(image_tensor.unsqueeze(0)).topk(topk)

    probs = torch.exp(predictions.values).tolist()[0]
    classes = []
    class_dict = model.class_to_idx
    def get_key(d, value):
        return next(key for key, val in d.items() if val == value)
    for idx in predictions.indices.tolist()[0]:
        classes.append(get_key(class_dict, idx))
    
    return probs, classes

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        show_image = True
        fig, ax = plt.subplots()
    else:
        show_image = False
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    image = image * std + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.set_title(title)
    
    return ax

def show_probs(img, probs, classes, cat_to_name_json):
    probs_reversed = probs[::-1]
    classes_reversed = classes[::-1]
    with open(cat_to_name_json, 'r') as f:
        cat_to_name = json.load(f)
    categories = []
    for c in classes_reversed:
        categories.append(cat_to_name[c])
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1 = imshow(img, ax1, title=categories[-1])
    ax1.set_axis_off()
    ax2 = fig.add_subplot(223)
    ax2.barh(np.arange(len(probs)), probs_reversed)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(probs)))
    ax2.set_yticklabels(categories, size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout
    plt.savefig('matplotlib.png')
    plt.show()