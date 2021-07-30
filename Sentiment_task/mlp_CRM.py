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
import copy
import argparse
import random
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device",type=int,default=0)
parser.add_argument("--epoch",type=int,default=50)
parser.add_argument("--lr",type=float,default=1e-3)
parser.add_argument("--batch_size",type=int,default=32)
parser.add_argument("--run_seed",type=int,default=4)
parser.add_argument("--save_folder",type=str,default="./mlp_CRM/")
parser.add_argument("--cf_model_folder", type = str, default= "./mlp_cf/lr:0.001/")
args = parser.parse_args()

train_path='./dataset/sentiment/combined/paired/train_paired.tsv'
valid_path='./dataset/sentiment/combined/paired/dev_paired.tsv'
test_path = './dataset/sentiment/combined/paired/test_paired.tsv'

device=torch.device(args.device)

train_total_path = './dataset/sentiment/combined/paired/train_paired.tsv'

data=pd.read_csv(train_total_path, sep = "\t")
text=[]
for item in data["Text"]:
    text.append(item)

vectorizer = TfidfVectorizer(max_features=20000)
vectorizer.fit(text)
vectorizer.vocabulary_

vector = vectorizer.transform(text)

def get_label(x):
    if x=="Negative":
        return 0
    elif x == "Positive":
        return 1

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

class cf_conv_linear_net (nn.Module):
    def __init__(self, hidde_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidde_channels, (3,1))
        self.fc = nn.Linear(hidde_channels * 2, 2, bias= False)
        # self.fc = nn.Linear(3,3)
    def forward(self, x):
        # x = x[:,:, 0:1, :]
        out = torch.flatten(self.conv1(x), start_dim= 1)
        # out = x.view(len(x),3)
        return self.fc(out)


def train_original(net, lr, train_iter, lamda):
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

def train_cf(net1, net2, cf_net, lr, train_iter):
    net1.eval()
    net2.train()
    cf_net.train()
    optimizer = optim.Adam([{"params":net2.parameters(),"lr":lr},
                            {"params":cf_net.parameters(),"lr":lr}])
    Loss = nn.CrossEntropyLoss()
    loss_total = 0
    batchs_in_epoch = 0
    total_count = 0
    total = 0
    for batch in tqdm(train_iter):
        loss = 0
        optimizer.zero_grad()
        label = batch[1].to(device)
        vectors = vectorizer.transform(batch[0])
        x = torch.Tensor(vectors.toarray()).to(torch.float32).to(device)
        for index in range(len(batch[0])):
            #the example is real
            if index%2 == 0:
                real_data = net1(x[index])
                cf_data = net1(x[index + 1])
                delta = net2(x[index + 1] - x[index])
                out = cf_net(torch.cat([real_data, cf_data, delta]).view(1,1,3,2))
                loss += Loss(out, torch.stack([label[int(index/2)]]))
        loss.backward()
        optimizer.step()
        loss_total += loss.item()/len(batch[0])
    return loss_total


#模型测试
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

def model_test_cf(net1, net2, cf_net, test_iter):
    net1.eval()
    net2.eval()
    cf_net.eval()
    correct=0
    total=0
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            data = batch[0]
            vectors = vectorizer.transform(data)
            data = torch.Tensor(vectors.toarray()).to(torch.float32).to(device)
            label= torch.tensor(batch[1]).to(device)
            for index in range(len(label)):
                total += 1
                real_out = net1(data[2 * index])
                cf_out = net1(data[2 * index + 1])
                delta_out = net2(data[2 * index + 1] - data[2 * index])
                _,out =torch.max(cf_net(torch.cat([real_out, cf_out, delta_out]).view(1,1,3,2)), 1)
                if out == label[index]:
                    correct += 1
    print("Accurary is:" + str(100*correct/total))
    return 100*correct/total


def mk_dir(save_path):
    try:
        os.mkdir(save_path)
    except:
        pass
    

def get_iter(train_path):
    count = 0
    batch_list = []
    train_data = pd.read_csv(train_path)
    train_indexs = [i for i in range(len(train_data))] 
    random.shuffle(train_indexs)
    for index in tqdm(train_indexs):
        if count == 0:
            label =torch.stack([torch.tensor(train_data["Sentiment"][index])])
            sent_list = [(train_data["Text"][index])]
            count += 1
        else :
            label = torch.cat([label, torch.stack([torch.tensor(train_data["Sentiment"][index])])])
            sent_list.append((train_data["Text"][index]))
            count += 1
        if count == args.batch_size:
            count = 0
            batch_list.append((sent_list, label))
    if count != 0:
        batch_list.append((sent_list, label))
    return batch_list

def get_cf_iter(train_path):
    count = 0
    batch_list = []
    train_data = pd.read_csv(train_path, sep = "\t")
    train_indexs = [i for i in range(int(len(train_data)/2))] 
    random.shuffle(train_indexs)
    for index in tqdm(train_indexs):
        if count == 0:
            label = torch.stack([torch.tensor(get_label(train_data["Sentiment"][index * 2]))])
            sent_list = [(train_data["Text"][index * 2])]
            sent_list.append((train_data["Text"][index * 2 + 1]))
            count += 2
        else :
            label = torch.cat([label, torch.stack([torch.tensor(get_label(train_data["Sentiment"][index * 2]))])])
            sent_list.append((train_data["Text"][index * 2]))
            sent_list.append((train_data["Text"][index * 2 + 1]))
            count += 2
        if count >= args.batch_size:
            count = 0
            batch_list.append((sent_list, label))
    if count != 0:
        batch_list.append((sent_list, label))
    return batch_list



train_iter = get_cf_iter(train_path)
valid_iter = get_cf_iter(valid_path)
test_iter = get_cf_iter(test_path)

train_acc_list = []
valid_acc_list = []
test_acc_list = []
mk_dir(args.save_folder)
saving_path = args.save_folder +"lr:"+str(args.lr) + "/"
mk_dir(saving_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

mk_dir(saving_path)
saving_path_orig = saving_path


seed = args.run_seed
max_val = 0
final_test_acc = 0
train_acc_list = []
valid_acc_list = []
test_acc_list = []
setup_seed(seed)
net1 = torch.load(args.cf_model_folder + str(seed) +"/max_val.pt", map_location= device)
net2 = copy.deepcopy(net1).to(device)
cf_net = cf_conv_linear_net(10)
cf_net = cf_net.to(device)
saving_path = saving_path_orig + "/" + str(seed) +"/"
mk_dir(saving_path)
result_path = saving_path + "/" 
mk_dir(result_path)
for i in range(10):
    train_iter = get_cf_iter(train_path)
    net1.train()
    print("==epoch:"+str(i)+"==")
    loss = train_cf(net1,net2,cf_net, args.lr, train_iter)
    train_acc = round(model_test_cf(net1,net2, cf_net, train_iter),2)
    valid_acc = round(model_test_cf(net1,net2, cf_net, valid_iter),2)
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)
    test_acc = round(model_test_cf(net1,net2, cf_net, test_iter),2)
    test_acc_list.append(test_acc)
    print("train_acc:"+str(train_acc))
    print("valid_max:"+str(max(valid_acc_list)))
    print("in that epcoh, the test_acc is:"+str(test_acc_list[valid_acc_list.index(max(valid_acc_list))]))
    with open(result_path + "/acc_list", 'a+') as f:
        f.write("epoch:"+str(i)+"  train_acc:"+str(train_acc)+" valid_acc:"+str(valid_acc)+" test_acc:"+str(test_acc) + " loss_total:" + str(loss)  +
            " valid_max:"+str(max(valid_acc_list))+" in that epcoh, the test_acc is:"+str(test_acc_list[valid_acc_list.index(max(valid_acc_list))])+ "\n")
    # torch.save(net1, result_path + "/" + "epoch_" + str(i) + ".pt")
    if valid_acc > max_val:
        torch.save(net2, result_path + "/max_net2.pt")
        torch.save(cf_net, result_path + "/max_cf_net.pt")
        max_val = valid_acc
        final_test_acc = test_acc
with open(saving_path_orig + "/final_acc", "a+") as f:
    f.write("randomseed: " + str(seed) + " val_acc:" + str(max_val) +" test_acc:" + str(final_test_acc) + "\n")


