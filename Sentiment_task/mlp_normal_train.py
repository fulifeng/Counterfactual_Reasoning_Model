from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import argparse
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.autograd import Variable
import argparse
import random
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device",type=int,default=1)
parser.add_argument("--epoch",type=int,default=50)
parser.add_argument("--lr",type=float,default=1e-3)
parser.add_argument("--batch_size",type=int,default=32)
parser.add_argument("--run_seed",type=int,default=4)
parser.add_argument("--save_folder",type=str,default="./mlp_normal_train/")
args = parser.parse_args()

train_path='./dataset/sentiment/orig/train.tsv'
valid_path='./dataset/sentiment/orig/dev.tsv'
test_path ='./dataset/sentiment/orig/test.tsv'

device=torch.device(args.device)

train_total_path = './dataset/sentiment/orig/train.tsv'

data=pd.read_csv(train_total_path, sep = "\t")
text=[]
for item in data["Text"]:
    text.append(item)

vectorizer = TfidfVectorizer(max_features=20000)
vectorizer.fit(text)
vectorizer.vocabulary_

vector = vectorizer.transform(text)


class my_net(nn.Module):
    def __init__(self, vec_dim, out_size):
        super(my_net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(vec_dim, out_size),
            nn.Softmax()
        )
    def forward(self, x):
        out = self.fc(x)
        return out


vector = vectorizer.transform(text)


def train_original(net, lr, train_iter):
    net.train()
    optimizer = optim.Adam([{"params":net.parameters(),"lr":lr}])
    Loss = nn.CrossEntropyLoss()
    loss_total=0
    batchs_in_epoch=0
    total_count=0
    total=0
    for batch in train_iter:
        data = batch[0]
        vectors = vectorizer.transform(data)
        label= torch.tensor(batch[1]).to(device)
        temp_1 = torch.Tensor(vectors.toarray()).to(torch.float32).to(device)
        output = net(temp_1)
        loss = Loss(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total+=loss.item()/args.batch_size
        batchs_in_epoch+=1
    return loss_total/batchs_in_epoch


def model_test(net, test_iter):
    net.eval()
    correct=0
    total=0
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            data = batch[0]
            vectors = vectorizer.transform(data)
            data = torch.Tensor(vectors.toarray()).to(torch.float32).to(device)
            label= torch.tensor(batch[1]).to(device)
            #print(sum(data))
            outputs = net(data)
            #print(outputs)
            _,predict = torch.max(outputs.data,1)
            #print(predict)
            total+=label.size(0)
            correct+=(predict==label).sum().item()
        print("Accurary is:" + str(100*correct/total))
        return 100*correct/total


def mk_dir(save_path):
    try:
        os.mkdir(save_path)
    except:
        pass

def get_label(text):
    if text == "Positive":
        return 1
    elif text == "Negative":
        return 0

def get_iter(train_path):
    count = 0
    batch_list = []
    train_data = pd.read_csv(train_path, sep= "\t")
    train_indexs = [i for i in range(len(train_data))] 
    random.shuffle(train_indexs)
    for index in tqdm(train_indexs):
        if count == 0:
            label =torch.stack([torch.tensor(get_label(train_data["Sentiment"][index]))])
            sent_list = [(train_data["Text"][index])]
            count += 1
        else :
            label = torch.cat([label, torch.stack([torch.tensor(get_label(train_data["Sentiment"][index]))])])
            sent_list.append((train_data["Text"][index]))
            count += 1
        if count == args.batch_size:
            count = 0
            batch_list.append((sent_list, label))
    if count != 0:
        batch_list.append((sent_list, label))
    return batch_list



train_iter = get_iter(train_path)
valid_iter = get_iter(valid_path)
test_iter = get_iter(test_path)


net1 = my_net(len(vectorizer.vocabulary_),2).to(device)


train_acc_list = []
valid_acc_list = []
test_acc_list = []





def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




seed = args.run_seed
max_val = 0
train_acc_list = []
valid_acc_list = []
test_acc_list = []
mk_dir(args.save_folder)
saving_path = args.save_folder +"lr:"+str(args.lr) + "/"
mk_dir(saving_path)
saving_path_orig = saving_path
saving_path = saving_path_orig + "/" + str(seed) +"/"
mk_dir(saving_path)
setup_seed(seed)
net1 = my_net(len(vectorizer.vocabulary_),2).to(device)
result_path = saving_path +"/"
mk_dir(result_path)
for i in range(args.epoch):
    train_iter = get_iter(train_path)
    net1.train()
    print("==epoch:"+str(i)+"==")
    loss = train_original(net1,args.lr,train_iter)
    train_acc = round(model_test(net1, train_iter),2)
    valid_acc = round(model_test(net1, valid_iter),2)
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)
    test_acc = round(model_test(net1,test_iter),2)
    test_acc_list.append(test_acc)
    print("train_acc:"+str(train_acc))
    print("valid_max:"+str(max(valid_acc_list)))
    print("in that epcoh, the test_acc is:"+str(test_acc_list[valid_acc_list.index(max(valid_acc_list))]))
    with open(result_path + "/acc_list", 'a+') as f:
        f.write("epoch:"+str(i)+"  train_acc:"+str(train_acc)+"  valid_acc:"+str(valid_acc)+"  test_acc:"+str(test_acc) + " loss_total:" + str(loss)  +
            " valid_max:"+str(max(valid_acc_list))+" in that epcoh, the test_acc is:"+str(test_acc_list[valid_acc_list.index(max(valid_acc_list))])+ "\n")
    # torch.save(net1, result_path + "/" + "epoch_" + str(i) + ".pt")
    if valid_acc > max_val:
        torch.save(net1, result_path + "/max_val.pt")
        max_val = valid_acc


