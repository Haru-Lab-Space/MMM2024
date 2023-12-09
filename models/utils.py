import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os
import copy
import cv2
import json
import transformers
from transformers import AutoImageProcessor, ViTModel
from PIL import Image, ImageMath
from transformers import AutoModel
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, BertTokenizer
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F
from ast import literal_eval
from torchvision.models import resnet50,SwinTransformer
import torchvision
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from transformers import AutoImageProcessor, AutoModelForImageClassification
import requests
from io import BytesIO
from transformers import RobertaTokenizer, RobertaModel
from focal_loss.focal_loss import FocalLoss
from sklearn.metrics import classification_report

CHECKPOINT_EXTENSION = '.pt'

def save_checkpoint(checkpoint_directory, epoch, model, optimizer, LOSS, checkpoint_name=None):
    """
    The checkpoint will be saved in `checkpoint_directory` with name `checkpoint_name`.
    If `checkpoint_name` is None, the checkpoint will be saved with name `next_checkpoint_name_id + epoch`.
    """
    if checkpoint_directory is not None:
        if checkpoint_name is None:
            checkpoint_name = f'{epoch}{CHECKPOINT_EXTENSION}'

        path = os.path.join(checkpoint_directory, checkpoint_name)
        print("path: "+str(path))
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': LOSS,
                    }, path)
def load_checkpoint(PATH, model, optimizer):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch, model, optimizer, loss

def save_report(pred_list, target_list, report_path):
    report = classification_report(target_list, pred_list, target_names=['NO', 'YES'], digits=6)
    with open(report_path, 'w') as f:
        f.write(report)

def read_json(parient_dir, name=None):
    # Opening JSON file
    if name == None:
        path = parient_dir
    else:
        path = os.path.join(parient_dir, str(name) + ".json")
    with open(path, 'r') as openfile:

        # Reading from json file
        json_object = json.load(openfile)
    return json_object
def write_json(dict, parient_dir, name=None):
    # Serializing json
    json_object = json.dumps(dict, indent=4)
    if name != None:
        path = os.path.join(parient_dir, str(name)+".json")
    else:
        path = parient_dir
    # Writing to sample.json
    with open(path, "w") as outfile:
        outfile.write(json_object)
# img_dir = '/content/drive/MyDrive/HaruLab/Dataset/Musti/test/img'
# json_file =
# values = json_file['pairs']

def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Open the image using Pillow
        image = Image.open(BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error loading image from URL: {e}")
        return None
def split_filename(filepath):
    # Split the filepath into the directory and file parts
    directory, full_filename = os.path.split(filepath)

    # Split the file part into the base name and extension
    base_name, file_extension = os.path.splitext(full_filename)

    return directory, base_name, file_extension
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
class Dataset_MMM(Dataset):
    def __init__(self, file_path,image_dir ,image_processor,tokenizer):
        super(Dataset_MMM, self).__init__()
        self.image_dir = image_dir
        data = read_json(file_path)
        if isinstance(data, dict):
            data = data['pairs']
        print(self.image_dir)
        self.data = []
        for obj in data:
          if os.path.exists(os.path.join(self.image_dir, obj['image'].rstrip())):
            self.data.append(obj)
          else:
            _, base_name, file_extension = split_filename(obj['image'].rstrip())
            if os.path.exists(os.path.join(self.image_dir, base_name + file_extension)):
              obj['image'] = base_name + file_extension
              self.data.append(obj)
            else:
              img = load_image_from_url(obj['image'].rstrip())
              obj['image'] = base_name + file_extension
              img.save(os.path.join(self.image_dir, base_name + file_extension))

              self.data.append(obj)
              print(base_name + file_extension)
        print("Len: " +str(len(self.data)))
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir,self.data[index]['image'])
        text = self.data[index]['text']
        target = self.data[index]['subtask1_label']
        if target == 'NO':
          target = 0
        else:
          target = 1
        # process_image
        img = Image.open(img_path)
        if img.mode == 'L':
          img = img.convert('RGB')
        img = self.image_processor(images = img, return_tensors="pt",padding=True)
        # process_text
        text = self.tokenizer(text,padding="max_length", max_length=197,truncation=True, return_tensors="pt")

        return img, text, target
    
class Dataset_Ex(Dataset):
    def __init__(self, file_path,image_dir ,image_processor,tokenizer):
        super(Dataset_Ex, self).__init__()
        self.image_dir = image_dir
        data = read_json(file_path)
        if isinstance(data, dict):
            data = data['pairs']
        print(self.image_dir)
        self.data = []
        for obj in data:
          if os.path.exists(os.path.join(self.image_dir, obj['image'].rstrip())):
            self.data.append(obj)
          else:
            _, base_name, file_extension = split_filename(obj['image'].rstrip())
            if os.path.exists(os.path.join(self.image_dir, base_name + file_extension)):
              obj['image'] = base_name + file_extension
              self.data.append(obj)
            else:
              img = load_image_from_url(obj['image'].rstrip())
              obj['image'] = base_name + file_extension
              img.save(os.path.join(self.image_dir, base_name + file_extension))

              self.data.append(obj)
              print(base_name + file_extension)
        print("Len: " +str(len(self.data)))
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir,self.data[index]['image'])
        text = self.data[index]['text']
        full_text = self.data[index]['text']
        target = self.data[index]['subtask1_label']
        if target == 'NO':
          target = 0
        else:
          target = 1
        # process_image
        img = Image.open(img_path)
        if img.mode == 'L':
          img = img.convert('RGB')
        img = self.image_processor(images = img, return_tensors="pt",padding=True)
        # process_text
        text = self.tokenizer(text,padding="max_length", max_length=197,truncation=True, return_tensors="pt")

        return img, text, target, full_text
    
def train(model, 
          train_loader,
          device, 
          epochs=100, 
          total_iterations_limit=None, 
          optimizer=None, 
          cur_epoch=0, 
          best_valid_loss=10000, 
          gamma=0, 
          alpha=0,
          checkpoint_directory="",
          report_path="",
          early_stopper=None
          ):

    assert checkpoint_directory != "", "Checkpoint path need to set!"
    assert report_path != "", "Report path need to set!"
    assert early_stopper != None, "early_stopper need to set!"
    assert gamma != 0, "early_stopper need to set!"
    assert alpha != 0, "early_stopper need to set!"

    
    history = {"loss": [],
                "acc": []}


    total_iterations = 0
    loss_fn = FocalLoss(gamma=gamma, weights=torch.tensor(alpha))
    for epoch in range(cur_epoch + 1, epochs):
        model.train()
        avg_acc = 0
        acc_sum = 0
        avg_loss = 0
        loss_sum = 0
        num_iterations = 0
        pred_list = []
        target_list = []

        data_iterator = tqdm(train_loader, desc=f'Epoch {epoch}')
        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            img = torch.squeeze(data[0]['pixel_values'],1)
            text = torch.squeeze(data[1]['input_ids'],1)
            target = torch.squeeze(data[2])
            # print(text.shape)
            img = img.to(device)
            text = text.to(device)
            target = target.to(device).to(torch.int64)
            optimizer.zero_grad()

            logits = model(img,text)
            loss = loss_fn(torch.sigmoid(logits), target)
            loss_sum += loss.item()

            pred = torch.round(torch.sigmoid(logits.detach()))
            correct_pred = (target == pred).float()
            acc_sum += (correct_pred.sum() / len(correct_pred)).cpu().item()


            pred_list.extend(pred.cpu().numpy())
            target_list.extend(target.cpu().numpy())


            data_iterator.set_postfix(loss=loss_sum / num_iterations)
            loss.backward()
            optimizer.step()

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return

        avg_loss = loss_sum / num_iterations
        avg_acc = acc_sum / num_iterations

        if avg_loss < best_valid_loss:
            best_valid_loss = avg_loss
            save_checkpoint(checkpoint_directory, epoch, model, optimizer, avg_loss)
            save_report(pred_list=pred_list, target_list=target_list, report_path= os.path.join(report_path, str(epoch) + '.txt'))
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        print("Accuracy: "+str(avg_acc))
        if early_stopper.early_stop(avg_loss):
                print("Early stop at epoch: "+str(epoch) + " with valid loss: "+str(avg_loss))
                break
    return history

def predict(model,device,dataloader,image_processor):
  list = []
  model.eval()

  data_iterator = tqdm(dataloader, desc=f'Evaluate')
  with torch.no_grad():
    for data in data_iterator:
      img = torch.squeeze(data[0]['pixel_values'],1)
      text = torch.squeeze(data[1]['input_ids'],1)
      target = torch.squeeze(data[2])
      # print(text.shape)
      img = img.to(device)
      text = text.to(device)
      target = target.to(device).to(torch.int64)

      logits = model(img,text)

      print("logits"+str(torch.sigmoid(logits)))
      pred = torch.round(torch.sigmoid(logits)).detach().cpu().numpy()
      list.extend(pred)
  return list