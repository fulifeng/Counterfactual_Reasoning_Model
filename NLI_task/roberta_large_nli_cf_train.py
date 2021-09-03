import torch
from transformers import BertModel,BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification
import argparse
import numpy as np
import sys
import torch.optim as optim
from torch import nn
# import spacy
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
parser.add_argument("--device", type= int, default= 0)

parser.add_argument("--train_file", type =str, default= "./dataset/NLI/all_combined/train.tsv")
parser.add_argument("--val_file", type =str, default= "./dataset/NLI/all_combined/dev.tsv")
parser.add_argument("--test_file", type=str, default= "./dataset/NLI/all_combined/test.tsv")
parser.add_argument("--orig_train_file", type =str, default= "./dataset/NLI/original/train.tsv")
parser.add_argument("--orig_val_file", type =str, default= "./dataset/NLI/original/dev.tsv")
parser.add_argument("--orig_test_file", type=str, default= "./dataset/NLI/original/test.tsv")
parser.add_argument("--revised_train_file", type =str, default= "./dataset/NLI/revised_combined/train.tsv")
parser.add_argument("--revised_val_file", type =str, default= "./dataset/NLI/revised_combined/dev.tsv")
parser.add_argument("--revised_test_file", type=str, default= "./dataset/NLI/revised_combined/test.tsv")
parser.add_argument("--run_seed", type = int, default= 4)
parser.add_argument("--lr", type=float, default= 1e-5)
parser.add_argument("--batchsize", type=int , default= 4)
parser.add_argument("--warm_up_rate", type=float, default=.1)
parser.add_argument("--epochs", type=int , default= 10)
parser.add_argument("--save_folder", type=str, default="./NLI_tasks/roberta_large_nli_cf/")
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


tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")


def get_label(text):
    if text == "neutral":
        return 1
    elif text == "contradiction":
        return 0
    elif text == "entailment":
        return 2

def create_batch(train_data, batchsize):
    count = 0
    batch_list = []
    train_indexs = [i for i in range(len(train_data))] 
    random.shuffle(train_indexs)
    for index in tqdm(train_indexs):
        if count == 0:
            label =torch.stack([torch.tensor(get_label(train_data["gold_label"][index]))])
            sent_list = [(train_data["sentence1"][index],train_data['sentence2'][index])]
            count += 1
        else :
            label = torch.cat([label, torch.stack([torch.tensor(get_label(train_data["gold_label"][index]))])])
            sent_list.append((train_data["sentence1"][index],train_data['sentence2'][index]))
            count += 1
        if count == batchsize:
            count = 0
            batch_list.append((label,sent_list))
    if count != 0:
        batch_list.append((label,sent_list))
    return batch_list


def isNan_2(a):
    return a != a

def mk_dir(path):
    try:
        os.makedirs(path)
    except:
        pass

train_data = pd.read_csv(args.train_file, sep= "\t")
val_data = pd.read_csv(args.val_file, sep ="\t")
test_data = pd.read_csv(args.test_file, sep = "\t")

orig_train_data = pd.read_csv(args.orig_train_file, sep= "\t")
orig_val_data = pd.read_csv(args.orig_val_file, sep ="\t")
orig_test_data = pd.read_csv(args.orig_test_file, sep = "\t")


batch_train = create_batch(train_data, args.batchsize)
batch_val = create_batch(val_data, args.batchsize)
batch_test = create_batch(test_data, args.batchsize)

orig_batch_val = create_batch(orig_val_data, args.batchsize)
orig_batch_test = create_batch(orig_test_data, args.batchsize)

def calc_warm_up(epochs, batch_train):
    total_steps = len(batch_train) * epochs
    warm_up_steps = args.warm_up_rate * total_steps
    return total_steps, warm_up_steps

total_steps, warm_up_steps = calc_warm_up(args.epochs, batch_train)

def model_test(batch_train, model):
    model = model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for index in tqdm(range(len(batch_train))):
            label = batch_train[index][0].to(device)
            encoder = tokenizer(batch_train[index][1], padding=True, truncation=True, max_length=512, return_tensors='pt' )
            output = model(encoder["input_ids"].to(device), encoder["attention_mask"].to(device))[0]
            # output = out_net(output)
            _,predict = torch.max(output,1)
            total+=label.size(0)
            correct += (predict == label).sum().item()
    return 100 * correct/total


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


total_orig_val_list = []
total_orig_test_list = []
seed = args.run_seed
total_val_max = 0
orig_test_max = 0
setup_seed(seed)
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli", num_labels = 3, from_tf = True).to(device)
_ = model.classifier.apply(weight_init)
optimizer2 = AdamW([{"params":model.parameters(),"lr":args.lr}])
scheduler = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps = warm_up_steps, num_training_steps = total_steps)
Loss = nn.CrossEntropyLoss()
for i in range(0, args.epochs): 
    loss_total = 0
    batch_train = create_batch(train_data, args.batchsize)
    for index in tqdm(range(len(batch_train))):
        encoder = tokenizer(batch_train[index][1], padding=True, truncation=True, max_length=512, return_tensors='pt')
        output = model(encoder["input_ids"].to(device), encoder["attention_mask"].to(device))[0]
        label = batch_train[index][0].to(device)
        loss = Loss(output, label)
        optimizer2.zero_grad()
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)
        optimizer2.step()
        scheduler.step()
        loss_total += loss.item()
    print("epoch:" + str(i))
    print(loss_total/len(batch_train))
    # batch_train = create_batch(train_data, 64)
    acc1 = model_test(batch_train, model)
    acc2 = model_test(batch_val, model)
    acc3 = model_test(batch_test, model)
    acc4 = model_test(orig_batch_val, model)
    acc5 = model_test(orig_batch_test, model)
    # print(loss_total/len(batch_train))
    print(acc1, acc2, acc3 , acc4, acc5)
    mk_dir(args.save_folder)
    final_save_folder = args.save_folder + "/" + str(seed) + "/"
    mk_dir(final_save_folder)
    if acc2 >  total_val_max:
        total_val_max = acc2
        orig_test_max = acc5
        torch.save(model, final_save_folder + "/roberta-large-mnli"  + ".pt")
    with open( final_save_folder + "/acc_out","a+") as f:
        f.write("epoch:" + str(i) + " train_acc:" + str(acc1) + " total_val_acc:" + str(acc2) + " orig_val_acc:" + str(acc4) + " total_test_acc:" + str(acc3) + " orig_test_acc:" + str(acc5) + "\n")
    if i == args.epochs - 1:
        with open(args.save_folder + "/" + "final_acc", "a+") as f:
            f.write("random seed:" + str(seed) + "total_val_acc: " + str(total_val_max) + "orig_test_acc: "+ str(orig_test_max) + "\n")


