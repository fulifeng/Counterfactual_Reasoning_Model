from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
#from pytorch_pretrained_bert import BertTokenizer, BertModel
from transformers import BertModel,BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import argparse
import numpy as np
import sys
import torch.optim as optim
from torch import nn
import pandas as pd
import numpy as np
# from sklearn.metrics import accuracy_score
from collections import namedtuple
from tqdm import tqdm
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--device", type= int, default= 3)

parser.add_argument("--train_file", type =str, default= "./dataset/sentiment/combined/train.tsv")
parser.add_argument("--val_file", type =str, default= "./dataset/sentiment/combined/dev.tsv")
parser.add_argument("--test_file", type=str, default= "./dataset/sentiment/combined/test.tsv")
parser.add_argument("--orig_train_file", type =str, default= "./dataset/sentiment/orig/train.tsv")
parser.add_argument("--orig_val_file", type =str, default= "./dataset/sentiment/orig/dev.tsv")
parser.add_argument("--orig_test_file", type=str, default= "./dataset/sentiment/orig/test.tsv")
parser.add_argument("--revised_train_file", type =str, default= "./dataset/sentiment/combined/paired/train_paired.tsv")
parser.add_argument("--revised_val_file", type =str, default= "./dataset/sentiment/combined/paired/dev_paired.tsv")
parser.add_argument("--revised_test_file", type=str, default= "./dataset/sentiment/combined/paired/test_paired.tsv")
parser.add_argument("--run_seed", type = int, default= 4)
parser.add_argument("--lr", type=float, default= 1e-3)
parser.add_argument("--batchsize", type=int , default= 4)
parser.add_argument("--warm_up_rate", type=float, default=.1)
parser.add_argument("--epochs", type=int , default= 20)
parser.add_argument("--save_folder", type=str, default ="./roberta_large_CRM/")
parser.add_argument("--cf_model_folder", type = str, default= "./roberta_large_cf_train/")
parser.add_argument("--log_name", type= str, default= "cf_inference_out.log")
parser.add_argument("--plot_name", type = str, default= "result_plot2.jpg")
args = parser.parse_args()

device = torch.device("cuda:"+str(args.device))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


tokenizer = AutoTokenizer.from_pretrained("roberta-large")

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



def get_label(text):
    if text == "Positive":
        return 1
    elif text == "Negative":
        return 0

def create_batch(train_data, batchsize):
    count = 0
    batch_list = []
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
        if count == batchsize:
            count = 0
            batch_list.append((label,sent_list))
    return batch_list

# orig_data = orig_train_data
# cf_data = revised_train_data


def calc_cf_sent_list(sent_list, model, tokenizer):
    model.eval()
    with torch.no_grad():
        real_out = model(**tokenizer(sent_list[0], padding=True, truncation=True, max_length=256, return_tensors='pt' ).to(device)).logits.detach()
        cf_out = model(**tokenizer(sent_list[1], padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)).logits.detach()
        delta_embed = model.roberta(**tokenizer(sent_list[1], padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)).last_hidden_state.detach()[:,:1,:]\
            - model.roberta(**tokenizer(sent_list[0], padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)).last_hidden_state.detach()[:,:1,:]
        # delta_out = model.classifier(delta_embed).detach()
        return delta_embed, [cf_out, real_out]



def create_batch_with_delta_cf(orig_data, cf_data, batchsize, model, tokenizer):
    model.eval()
    with torch.no_grad():
        count  = 0
        batch_list = []
        data_indexs = [i for i in range(len(orig_data))]
        random.shuffle(data_indexs)
        for index in tqdm(data_indexs):
            if count == 0:
                label =torch.stack([torch.tensor(get_label(cf_data["Sentiment"][index * 2]))])
                sent_list = [cf_data["Text"][index * 2]]
                sent_list.append(cf_data["Text"][index * 2 + 1])
                delta_embed, output =calc_cf_sent_list(sent_list, model, tokenizer)
                delta_embed_list = [delta_embed]
                output = torch.cat(output)
                output_list = [output]
                count = count + 1
            else:
                # label =torch.stack([torch.tensor(get_label(orig_data["gold_label"][index]))])
                sent_list = [cf_data["Text"][index * 2]]
                sent_list.append(cf_data["Text"][index * 2 + 1])
                delta_embed, output =calc_cf_sent_list(sent_list, model, tokenizer)
                delta_embed_list.append(delta_embed)
                output = torch.cat(output)
                output_list.append(output)
                label = torch.cat([label, torch.stack([torch.tensor(get_label(cf_data["Sentiment"][index * 2]))])])
                count = count + 1  
            if count == batchsize:
                count = 0
                # embed_list = torch.stack([torch.stack([j]) for j in embed_list])
                batch_list.append((label, delta_embed_list, output_list))
        if count != 0:
            # embed_list = torch.stack([torch.stack([j]) for j in embed_list])
            batch_list.append((label, delta_embed_list, output_list))
    return batch_list

def isNan_2(a):
    return a != a

def mk_dir(path):
    try:
        os.mkdir(path)
    except:
        pass

train_data = pd.read_csv(args.train_file, sep= "\t")
val_data = pd.read_csv(args.val_file, sep ="\t")
test_data = pd.read_csv(args.test_file, sep = "\t")

orig_train_data = pd.read_csv(args.orig_train_file, sep= "\t")
orig_val_data = pd.read_csv(args.orig_val_file, sep ="\t")
orig_test_data = pd.read_csv(args.orig_test_file, sep = "\t")

revised_train_data = pd.read_csv(args.revised_train_file, sep= "\t")
revised_val_data = pd.read_csv(args.revised_val_file, sep ="\t")
revised_test_data = pd.read_csv(args.revised_test_file, sep = "\t")


def model_test(batch_train, classifier, cf_net):
    cf_net = cf_net.eval()
    classifier = classifier.eval()
    correct=0
    total=0
    with torch.no_grad():
        for index in tqdm(range(len(batch_train))):
            label = batch_train[index][0].to(device)
            # encoder = tokenizer(batch_train[index][1], padding=True, truncation=True, max_length=512, return_tensors='pt' )
            out = classifier(torch.cat(batch_train[index][1])).view(len(label),2)
            # output = model(**tokenizer(batch_train[index][1], padding=True, truncation=True, max_length=256, return_tensors='pt' ))
            output = cf_net(torch.cat([torch.stack(batch_train[index][2]).view(len(label),2,2),out.view(len(label),1,2)], dim=1).view(len(label),1,3,2))
            # output = out_net(output)
            _,predict = torch.max(output,1)
            total+=label.size(0)
            correct += (predict == label).sum().item()
    return 100 * correct/total



def shuffle_from_bs_1(batch_train_bs_1, batchsize):
    batch_train_bs = copy.deepcopy(batch_train_bs_1)
    count = 0
    batch_list = []
    index_list = [i for i in range(len(batch_train_bs))]
    random.shuffle(index_list)
    for index in index_list:
        item = batch_train_bs[index]
        if count == 0:
            label_1 = item[0]
            delta_1 = item[1]
            out_1 = item[2]
            count += 1
        else:
            label_1 = torch.cat([label_1, item[0]])
            delta_1 += item[1]
            out_1 += item[2]
            count += 1
        if count >= batchsize:
            batch_list.append((label_1, delta_1, out_1))
            count = 0
    if count != 0:
        batch_list.append((label_1, delta_1, out_1))
    return batch_list

seed = args.run_seed
setup_seed(seed)
model = torch.load(args.cf_model_folder + str(seed) + "/roberta-large.pt", map_location = device)
classifier = copy.copy(model.classifier).to(device)
cf_net = cf_conv_linear_net(10).to(device)
optimizer = optim.Adam([{"params":cf_net.parameters()},
                        {"params":classifier.parameters()}])
Loss = nn.CrossEntropyLoss()
batch_train_bs_1 = create_batch_with_delta_cf(orig_train_data, revised_train_data, 1, model, tokenizer)
batch_val = create_batch_with_delta_cf(orig_val_data, revised_val_data, args.batchsize, model, tokenizer)
batch_test = create_batch_with_delta_cf(orig_test_data, revised_test_data, args.batchsize, model, tokenizer)
acc_train_list = []
acc_val_list = []
acc_test_list = []
max_val_acc = 0
final_test_acc = 0 
mk_dir(args.save_folder)
saving_folder = args.save_folder + "/" + str(seed) + "/"
mk_dir(saving_folder)
for i in range(0, args.epochs): 
    print("epoch:" + str(i))
    loss_total = 0
    batch_train = shuffle_from_bs_1(batch_train_bs_1, args.batchsize)
    with open(saving_folder + "/" + args.log_name,"a+") as f:
        if i == 0:
            f.write("settings:\n")
            f.write("lr:" + str(args.lr) + "\n")
            f.write("net_struc:" + "\n")
            print(cf_net, file=f)
            print(classifier, file=f)
            acc1 = model_test(batch_train, classifier, cf_net)
            acc2 = model_test(batch_val, classifier, cf_net)
            acc3 = model_test(batch_test,classifier, cf_net)
    for index in tqdm(range(len(batch_train))):
        # encoder = tokenizer(batch_train[index][1], padding=True, truncation=True, max_length=512, return_tensors='pt' )
        label = batch_train[index][0].to(device)
        out = classifier(torch.cat(batch_train[index][1])).view(len(label),2)
        output = cf_net(torch.cat([torch.stack(batch_train[index][2]).view(len(label),2,2),out.view(len(label),1,2)], dim=1).view(len(label),1,3,2))
        # output = cf_net(batch_train[index][1])
        loss = Loss(output, label)
        # _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        loss_total += loss.item()
    print(loss_total/len(batch_train))
    # batch_train = create_batch(train_data, 64)
    acc1 = model_test(batch_train, classifier, cf_net)
    acc2 = model_test(batch_val, classifier, cf_net)
    acc3 = model_test(batch_test, classifier, cf_net)
    acc_train_list.append(acc1)
    acc_val_list.append(acc2)
    acc_test_list.append(acc3)
    # acc4 = model_test(orig_batch_val, model)
    # acc5 = model_test(orig_batch_test, model)
    # print(loss_total/len(batch_train))
    print(acc1, acc2, acc3)
    if acc2 > max_val_acc:
        max_val_acc = acc2
        final_test_acc = acc3
        max_cf_net = cf_net
        max_classifier = classifier
        torch.save(cf_net, saving_folder + "/max_cf_net.pt")
        torch.save(classifier, saving_folder + "/max_classifier.pt")
    # torch.save(model, args.save_folder + "roberta-large-mnli" + "save_epoch_" + str(i) + ".pt")
    with open(saving_folder + "/" + args.log_name,"a+") as f:
        f.write("epoch:" + str(i) + " train_acc:" + str(acc1) + " val_acc:" + str(acc2) + " test_acc:" + str(acc3) + "\n")
    mk_dir(saving_folder + str(i) + "epoch")
    torch.save(classifier, saving_folder + str(i) + "epoch" + "/classifier.pt")
    torch.save(cf_net, saving_folder + str(i) + "epoch" + "/cf_net.pt")
x = [i for i in range(len(acc_train_list))]
p1 = plt.plot(x, acc_train_list, "b", marker = "o", label = "train")
p2 = plt.plot(x, acc_val_list, "g", marker = "v", label = "val")
p3 = plt.plot(x, acc_test_list, "y", marker = "^", label = "test")
plt.xlabel("epochs")
plt.ylabel("acc")
plt.title("cf_net result")
# plt.legend([p1,p2,p3], ["train", "val", "test"])
# plt.title("cf_net result")
plt.legend(labels = ["train", "val", "test"])
plt.savefig(saving_folder + args.plot_name)
plt.cla()
# torch.save(saving_folder + "/max_cf_net.pt", cf_net)
# torch.save(saving_folder + "/max_classifier.pt", cf_net)
with open(args.save_folder + "/final_acc", "a+") as f:
    f.write("ramdom seed:" + str(seed) + " max_val_acc:" + str(max_val_acc) + " test_acc:" + str(final_test_acc) + "\n")


