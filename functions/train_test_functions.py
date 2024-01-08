"""
Functions to handle the training and testing of the models
"""

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


def init_weights(m):
  if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
    nn.init.normal_(m.weight,0,0.01)
    nn.init.zeros_(m.bias)

def train_one_epoch(model,epoch_index,loader,device,loss_function,optimizer):
  running_loss=0

  for i, data in enumerate(loader):

    inputs,labels=data #get the minibatch
    if device.type == 'cuda':
      inputs, labels = inputs.cuda(), labels.cuda()

    outputs=model(inputs) #forward pass

    loss=loss_function(outputs,labels) #compute loss
    running_loss+=loss.item() #sum up the loss for the minibatches processed so far

    optimizer.zero_grad() #reset gradients
    loss.backward() #compute gradient
    optimizer.step() #update weights

  return running_loss/(i+1) # average loss per minibatch


def train_model(model,train_loader,validation_loader,loss_function,optimizer,EPOCHS,device):
  best_validation_loss=np.inf

  train_losses = []
  validation_losses = []
  validation_accuracies = []

  start_time=time.time()
  for epoch in range(EPOCHS):
    print('EPOCH{}:'.format(epoch+1))

    model.train(True)
    train_loss=train_one_epoch(model,epoch,train_loader,device,loss_function,optimizer) ##train for each epoch

    running_validation_loss=0.0

    model.eval()

    with torch.no_grad(): # Disable gradient computation and reduce memory consumption
      correct=0
      total=0
      for i,vdata in enumerate(validation_loader):
        vinputs,vlabels=vdata
        voutputs=model(vinputs)
        _,predicted=torch.max(voutputs.data,1)
        vloss=loss_function(voutputs,vlabels)
        running_validation_loss+=vloss
        total+=vlabels.size(0)
        correct+=(predicted==vlabels).sum().item()
    validation_loss=running_validation_loss/(i+1)
    validation_acc = 100*correct/total
    print(f'LOSS train: {train_loss} validation: {validation_loss} | validation_accuracy: {validation_acc}% ')

    if validation_loss<best_validation_loss: #save the model if it's the best so far
      timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
      best_validation_loss=validation_loss
      model_path='model_{}_{}'.format(timestamp,epoch)
      torch.save(model.state_dict(),model_path)

    train_losses.append(train_loss)
    validation_losses.append(validation_loss)
    validation_accuracies.append(validation_acc)

  end_time=time.time()

  plt.plot(train_losses, color='tab:red', linewidth=3, label='train loss')
  validation_losses_np = torch.stack(validation_losses).cpu().numpy() #move validation losses to cpu to plot with matplotlib
  plt.plot(validation_losses_np, color='tab:green', linewidth=3, label='validation loss')
  plt.xlabel('Epoch')
  plt.ylabel('CE loss')

  ax_right = plt.gca().twinx()
  #validation_accuracies_np = torch.stack(validation_accuracies).cpu().numpy() #move validation accuracies to cpu to plot with matplotlib
  ax_right.plot(validation_accuracies, color='tab:green', linestyle='--', label='validation accuracy')
  ax_right.set_ylabel('accuracy (%)')

  plt.gcf().legend(ncol=3)
  plt.gcf().set_size_inches(6, 3)

  print(f"Time: {end_time-start_time}")
  return model_path


def test_model(model,model_path,test_loader,test_set,device):
  #load the best model and evaluate performance on the test set

  model.to(device)
  model.load_state_dict(torch.load(model_path))
  model.eval()

  correct=0
  total=0

  y_pred = []
  y_true = []

  with torch.no_grad():
    for data in test_loader:
      images,labels=data
      if device.type == 'cuda':
        images, labels = images.cuda(), labels.cuda()
      outputs=model(images)
      _,predicted=torch.max(outputs.data,1)
      total+=labels.size(0)
      correct+=(predicted==labels).sum().item()
      y_pred.extend(predicted) # Save Prediction
      y_true.extend(labels)  # Save Truth

  print(f"Accuracy of the network on the test images: {100*correct/total}%")

  # Build confusion matrix
  classes=test_set.classes
  cf_matrix = confusion_matrix(torch.stack(y_pred).cpu().numpy(), torch.stack(y_true).cpu().numpy())
  df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                      columns = [i for i in classes])
  plt.figure(figsize = (5,5))
  sn.heatmap(df_cm)

def test_svm(model,test_features,test_labels, test_set):
    accuracy=model.score(test_features,test_labels)
    print(f"Accuracy on test images is {accuracy}")
    predictions=model.predict(test_features)
    classes=test_set.classes
    cf_matrix = confusion_matrix(np.stack(predictions), np.stack(test_labels))
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                      columns = [i for i in classes])
    plt.figure(figsize = (5,5))
    sn.heatmap(df_cm)