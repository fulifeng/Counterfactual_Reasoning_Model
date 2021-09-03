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
from sklearn.metrics import accuracy_score
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
parser.add_argument("--lr", type=float, default= 1e-3)
parser.add_argument("--batchsize", type=int , default= 8)
parser.add_argument("--epochs", type=int , default= 20)
parser.add_argument("--run_seed", type = int, default= 4)
parser.add_argument("--save_folder", type=str, default ="./NLI_tasks/roberta_large_nli_EF/")
parser.add_argument("--log_name", type= str, default= "cf_inference_out.log")
parser.add_argument("--plot_name", type = str, default= "result_plot.jpg")
parser.add_argument("--cf_model_folder", type = str, default="./NLI_tasks/roberta_large_nli_cf/")
args = parser.parse_args()


device = torch.device("cuda:"+str(args.device))


tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")


class cf_conv_linear_net (nn.Module):
    def __init__(self, hidde_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidde_channels, (3,1))
        self.fc = nn.Linear(hidde_channels * 3, 3)
        # self.fc = nn.Linear(3,3)
    def forward(self, x):
        # x = x[:,:, 0:1, :]
        out = torch.flatten(self.conv1(x), start_dim= 1)
        # out = x.view(len(x),3)
        return self.fc(out)

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
    return batch_list


# 保留delta项
def create_batch_with_delta_cf(orig_data, cf_data, batchsize, model, tokenizer):
    model.eval()
    with torch.no_grad():
        count  = 0
        batch_list = []
        data_indexs = [i for i in range(len(orig_data))]
        random.shuffle(data_indexs)
        for index in tqdm(data_indexs):
            if count == 0:
                label =torch.stack([torch.tensor(get_label(orig_data["gold_label"][index]))])
                sent_list = [(orig_data["sentence1"][index],orig_data['sentence2'][index])]
                for i in range(4*index, 4 * index + 4):
                    sent_list.append((cf_data["sentence1"][i],cf_data['sentence2'][i]))
                delta_embed, output =calc_cf_sent_list(sent_list, model, tokenizer)
                delta_embed_list = [delta_embed]
                output = torch.cat(output)
                output_list = [output]
                count = count + 1
            else:
                # label =torch.stack([torch.tensor(get_label(orig_data["gold_label"][index]))])
                sent_list = [(orig_data["sentence1"][index],orig_data['sentence2'][index])]
                for i in range(4*index, 4 * index + 4):
                    sent_list.append((cf_data["sentence1"][i],cf_data['sentence2'][i]))
                delta_embed, output =calc_cf_sent_list(sent_list, model, tokenizer)
                delta_embed_list.append(delta_embed)
                output = torch.cat(output)
                output_list.append(output)
                label = torch.cat([label, torch.stack([torch.tensor(get_label(orig_data["gold_label"][index]))])])
                count = count + 1  
            if count == batchsize:
                count = 0
                # embed_list = torch.stack([torch.stack([j]) for j in embed_list])
                batch_list.append((label, delta_embed_list, output_list))
        if count != 0:
            # embed_list = torch.stack([torch.stack([j]) for j in embed_list])
            batch_list.append((label, delta_embed_list, output_list))
    return batch_list


def calc_cf_sent_list(sent_list, model, tokenizer):
    model.eval()
    with torch.no_grad():
        real_out = model(**tokenizer(sent_list[:1], padding=True, truncation=True, max_length=512, return_tensors='pt' ).to(device)).logits.detach()
        cf_out =torch.stack([torch.mean(model(**tokenizer(sent_list[1:5], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)).logits.detach(), dim = 0)])
        delta_embed = torch.mean(model.roberta(**tokenizer(sent_list[1:5], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)).last_hidden_state.detach()[:,:1,:], dim = 0).view(1,1,-1)\
            - model.roberta(**tokenizer(sent_list[:1], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)).last_hidden_state.detach()[:,:1,:]
        # delta_out = model.classifier(delta_embed).detach()
        return delta_embed, [cf_out, real_out]


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
            out= classifier(torch.cat(batch_train[index][1])).view(len(label),1,3)
            output = cf_net(torch.cat([torch.stack(batch_train[index][2]).view(len(label),2,3),out.view(len(label),1,3)], dim=1).view(len(label),1,3,3))
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = args.run_seed
max_val_acc = 0
final_test_acc = 0
setup_seed(seed)
model = torch.load(args.cf_model_folder + "/" +str(seed) + "/roberta-large-mnli.pt", map_location= device)
batch_train_bs_1 = create_batch_with_delta_cf(orig_train_data, revised_train_data, 1, model, tokenizer)
batch_val = create_batch_with_delta_cf(orig_val_data, revised_val_data, args.batchsize, model, tokenizer)
batch_test = create_batch_with_delta_cf(orig_test_data, revised_test_data, args.batchsize, model, tokenizer)
classifier = copy.deepcopy(model.classifier).to(device)
cf_net = cf_conv_linear_net(10).to(device)
optimizer = optim.Adam([{"params":cf_net.parameters()},
                        {"params":classifier.parameters()}], lr= args.lr)
Loss = nn.CrossEntropyLoss()
acc_train_list = []
acc_val_list = []
acc_test_list = []
mk_dir(args.save_folder)
final_saving_folder = args.save_folder + "/" + str(seed) + "/"
mk_dir(final_saving_folder)
for i in range(0, args.epochs): 
    print("epoch:" + str(i))
    loss_total = 0
    batch_train = shuffle_from_bs_1(batch_train_bs_1, args.batchsize)
    with open(final_saving_folder + "/" + args.log_name,"a+") as f:
        if i == 0:
            f.write("settings:\n")
            f.write("lr:" + str(args.lr) + "\n")
            f.write("net_struc:" + "\n")
            print(cf_net, file=f)
            print(classifier, file=f)
            acc1 = model_test(batch_train, classifier, cf_net)
            acc2 = model_test(batch_val, classifier, cf_net)
            acc3 = model_test(batch_test,classifier, cf_net)
            # acc_train_list.append(acc1)
            # acc_val_list.append(acc2)
            # acc_test_list.append(acc3)
            # f.write("before optim:" + str(i) + " train_acc:" + str(acc1) + " total_val_acc:" + str(acc2) + " total_test_acc:" + str(acc3) + "\n")
    for index in tqdm(range(len(batch_train))):
        # encoder = tokenizer(batch_train[index][1], padding=True, truncation=True, max_length=512, return_tensors='pt' )
        label = batch_train[index][0].to(device)
        out = classifier(torch.cat(batch_train[index][1])).view(len(label),1,3)
        output = cf_net(torch.cat([torch.stack(batch_train[index][2]).view(len(label),2,3),out.view(len(label),1,3)], dim=1).view(len(label),1,3,3))
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
    # torch.save(model, args.save_folder + "roberta-large-mnli" + "save_epoch_" + str(i) + ".pt")
    with open(final_saving_folder + "/" + args.log_name,"a+") as f:
        if i == 0:
            f.write("settings:\n")
            f.write("lr:" + str(args.lr) + "\n")
            f.write("net_struc:" + "\n")
            print(cf_net, file=f)
        f.write("epoch:" + str(i) + " train_acc:" + str(acc1) + " val_acc:" + str(acc2) + " test_acc:" + str(acc3) + "\n")
    if acc2 > max_val_acc:
        max_val_acc = acc2
        final_test_acc = acc3
        torch.save(classifier, final_saving_folder + "/classifier.pt")
        torch.save(cf_net, final_saving_folder + "/cf_net.pt")
with open(args.save_folder + "/final_acc", "a+") as f:
    f.write("random seed:" + str(seed) + "max_val_acc: " + str(max_val_acc) + " final_test_acc: " + str(final_test_acc) + "\n")
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
plt.savefig(final_saving_folder + args.plot_name)
plt.cla()
# end

