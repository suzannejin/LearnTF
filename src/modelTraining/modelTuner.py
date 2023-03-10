import sys 
sys.path.insert(1, "/Users/mgrapotte/LabWork/LearnTF/" )

import numpy as np
import os
import ray.tune as tune
from ray.tune.schedulers import ASHAScheduler
from maTransformerV1 import *
from torch.utils.data import random_split, Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

from src.datasetSimulation.TFFamilyClass import *
from src.utils.preProcessing import *

class parseData(Dataset):

    def __init__(self, data):
        self.dna = np.stack([encode_dna(d) for d in data["dna_seq"]])
        self.dna = torch.from_numpy(self.dna).reshape(self.dna.shape[0],1,self.dna.shape[1],self.dna.shape[2]).float()
        self.prot = np.stack([encode_protein(p) for p in data["prot_seq"]])
        self.prot = torch.from_numpy(self.prot).reshape(self.prot.shape[0],1,self.prot.shape[1],self.prot.shape[2]).float()
        self.label = data["label"].values

    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, idx):
        return self.dna[idx,:,:,:], self.prot[idx,:,:,:], self.label[idx]

class convDNA(nn.Module):
    def __init__(self, config):
        super(convDNA, self).__init__()
        self.conv1= nn.Conv2d(1,1,(config["alphabet"], config["filterSize"]), bias=False)
        self.linear= nn.Linear(config["inputSize"]-config["filterSize"]+1,1)
    
    def forward(self, dna):
        batch_size  = dna.shape[0]
        dna_conv = self.conv1(dna)       
        dna_conv = F.relu(dna_conv)      
        dna_conv = dna_conv.squeeze()
        dna_conv = self.linear(dna_conv)  
        return(dna_conv)

def loadDummyData(batch_size, label1=5):
    tf_object = TfFamily("data/raw_data/PWM.txt", "data/raw_data/prot_seq.txt")
    data = SimulatedData(tf_object, n=100)
    seqs = data.dummy_data['dna_seq'].values
    label = []

    # check if the sequence contains the motif
    for i in range(len(seqs)):
        if 'A' in seqs[i]:
            label.append(label1)
        else:
            label.append(0)
    
    data.dummy_data['label'] = label
    trainloader = DataLoader(parseData(data.dummy_data), batch_size=batch_size, shuffle=True)
    return trainloader


def train(config, checkpoint_dir=None, data_dir=None):



    # load data
    trainloader = loadDummyData(batch_size=config["batch_size"])
    
    # initialize model 
    net = convDNA(config["modelconfig"])

    # send model to the right device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    # setup optimizer and loss according to config
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=config["lr"])

    if config["loss"] == "mse":
        criterion = nn.MSELoss()
    elif config["loss"] == "mae":
        criterion = nn.L1Loss()
    elif config["loss"] == "smooth_l1":
        criterion = nn.SmoothL1Loss()

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # train model
    for epoch in range(50):
        train_loss = 0.0
        train_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            dna, prot, labels = data
            dna, prot, labels = dna.to(device), prot.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(dna)
            loss = criterion(outputs, labels.view(-1,1).float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(
                        (net.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=(train_loss / train_steps)) #TODO , accuracy=spearmanr(true, pred)[0])   


def roc_accuracy(labels, pred, pos_label):
    return roc_auc_score(labels, pred, pos_label)


def main():
    """Main function
    Functions defines a config file that contains parameter models and schedulers/loss functions. 
    Then initializes a ray cluster and runs the experiment.
    Ends by printing the best parameters. 

    Input : None
    Output : None
    
    """
    # define hyperparameters

    config = {
        'modelconfig' : {
            "filterSize": tune.sample_from(lambda _: np.random.randint(2, 20)),
            'alphabet': 4,
            'inputSize': 100 
        },
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "loss": tune.choice(["mse", "mae", "smooth_l1"]),
        "optimizer": tune.choice(["adam", "sgd"]),
    }    

    # define search algorithm
    scheduler = ASHAScheduler(
        max_t=10,  # number of epochs to train per trial
        grace_period=1,
        reduction_factor=2)

    # run experiment
    analysis = tune.run(
        train,
        metric="loss",
        mode="min",
        config=config,
        num_samples=10,
        scheduler=scheduler)

    # print best hyperparameters
    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == "__main__":
    main()



