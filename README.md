# Counterfactual_Reasoning_Model
The source code of "Empowering Language Understanding with Counterfactual Reasoning" (ACL'21)

# Counterfactual_Reasoning_Model

 The source code of "Empowering Language Understanding with Counterfactual Reasoning" (ACL'21) based on PyTorch

## Requirements

python3 (3.7.9)

torch==1.7.0 (cuda==10.2) CUDA Version 10.2.89

tensorflow==2.4.0

tqdm==4.54.1

scikit-learn==0.23.2

transformers==4.1.1

pandas==1.1.5

matplotlib==3.3.3

numpy==1.19.4

future==0.18.2

dataclasses==0.6

If these are not enough to reproduce the environment, we put the "requirements.txt" in the code which you can use to  create an environment by using: (python==3.7.9, pip==20.2.4, CUDA Version 10.2.89)

```
pip install -r requirements.txt
```



## Datasets

We use dataset in Paper " LEARNING THE DIFFERENCE THAT MAKES A DIFFERENCE WITH COUNTERFACTUALLY-AUGMENTED DATA " (ICLR'20), you can find this dataset in [[Dataset](https://github.com/dkaushik96/counterfactually-augmented-data)].

You can move the two folder (NLI, sentiment) in the [[Dataset](https://github.com/dkaushik96/counterfactually-augmented-data)] to ./dataset/ in our code for our code's default setting is data in this folder.

## Sentiment Task

### Parameters

Key Parameters in codes in Sentiment task:

--device: which GPU you would like to use while running the code

--run_seed: the random seed use to initialize learning model and random number generator.

-- save_folder: this is the file for saving trained models and result logs

--cf_model_folder: the path for trained +CF model, using in CRM

params like "--train_file", "--val_file" and so on are the paths of datasets using in the code, if you have already put the dataset in "./dataset" as "./dataset/sentiment/", you do not need to  modify them for the default value have already working very well.

### Commands

In this section, we will use random seed == 4 as example to show how to use our code.

+ First of all, cd to the dir "Counterfactual_Reasoning_Model"

#### MLP

##### mlp normal train

```
python Sentiment_task/mlp_normal_train.py --run_seed 4 --device 0 --lr 1e-3 --epoch 50 --save_folder ./mlp_normal_train/
```

While training, the result will be shown on the screen and write into file "acc_list" under save_folder

If you have same environment as me, the first five results would be:

```
epoch:0  train_acc:95.25  valid_acc:84.08  test_acc:81.56 loss_total:0.021539603049556415 valid_max:84.08 in that epcoh, the test_acc is:81.56
epoch:1  train_acc:93.61  valid_acc:86.12  test_acc:82.17 loss_total:0.021059105655661336 valid_max:86.12 in that epcoh, the test_acc is:82.17
epoch:2  train_acc:94.67  valid_acc:85.71  test_acc:83.4 loss_total:0.02059613133746165 valid_max:86.12 in that epcoh, the test_acc is:82.17
epoch:3  train_acc:95.2  valid_acc:86.94  test_acc:84.43 loss_total:0.020152651635860955 valid_max:86.94 in that epcoh, the test_acc is:84.43
epoch:4  train_acc:95.78  valid_acc:86.94  test_acc:85.04 loss_total:0.019709763807003147 valid_max:86.94 in that epcoh, the test_acc is:84.43
```

and the final result is:

```
epoch:49  train_acc:99.82  valid_acc:90.2  test_acc:86.68 loss_total:0.011687375622353068 valid_max:90.2 in that epcoh, the test_acc is:87.3
```

##### mlp CF train:

```
python Sentiment_task/mlp_cf_train.py --device 0 --epoch 50 --lr 1e-3 --run_seed 4 --save_folder ./mlp_cf/ 
```

While training, the result will be shown on the screen and write into file "acc_list" under save_folder

If you have same environment as me, the first five results would be:

```
epoch:0  train_acc:85.79 total_valid_acc:87.35 orig_test_acc:84.63 loss_total:0.02165383753305841 total_valid_max:87.35 in that epcoh, the orig_test_acc is:84.63
epoch:1  train_acc:88.78 total_valid_acc:87.76 orig_test_acc:84.43 loss_total:0.021384278103430694 total_valid_max:87.76 in that epcoh, the orig_test_acc is:84.43
epoch:2  train_acc:88.81 total_valid_acc:89.59 orig_test_acc:84.84 loss_total:0.021120074596778254 total_valid_max:89.59 in that epcoh, the orig_test_acc is:84.84
epoch:3  train_acc:88.66 total_valid_acc:89.18 orig_test_acc:85.25 loss_total:0.020860580065957854 total_valid_max:89.59 in that epcoh, the orig_test_acc is:84.84
epoch:4  train_acc:89.95 total_valid_acc:88.78 orig_test_acc:85.86 loss_total:0.020601281913641457 total_valid_max:89.59 in that epcoh, the orig_test_acc is:84.84
```

the final result is:

```
epoch:49  train_acc:95.14 total_valid_acc:89.8 orig_test_acc:84.84 loss_total:0.014604703479698885 total_valid_max:90.2 in that epcoh, the orig_test_acc is:85.04
```

##### mlp CRM

After running the code for mlp cf, you can run this for CRM

```
python Sentiment_task/mlp_CRM.py --device 0 --epoch 50 --lr 1e-3 --run_seed 4 --save_folder ./mlp_CRM/ --cf_model_folder ./mlp_cf/lr:0.001/
```

by setting this, the model will use cf_model in "./mlp_cf/lr:0.001/" + args.run_seed + "/max_val.pt", which has already been trained in mlp +cf.

While training, the result will be shown on the screen and write into file "acc_list" under save_folder

If you have same environment as me, the first five results would be:

```
epoch:0  train_acc:97.83 valid_acc:95.92 test_acc:95.9 loss_total:26.68698797984557 valid_max:95.92 in that epcoh, the test_acc is:95.9
epoch:1  train_acc:98.42 valid_acc:96.33 test_acc:97.75 loss_total:14.354043466800993 valid_max:96.33 in that epcoh, the test_acc is:97.75
epoch:2  train_acc:98.95 valid_acc:97.96 test_acc:98.36 loss_total:6.978138490834019 valid_max:97.96 in that epcoh, the test_acc is:98.36
epoch:3  train_acc:99.06 valid_acc:98.37 test_acc:98.57 loss_total:3.903942043723708 valid_max:98.37 in that epcoh, the test_acc is:98.57
epoch:4  train_acc:99.12 valid_acc:97.96 test_acc:98.36 loss_total:2.6727270136206327 valid_max:98.37 in that epcoh, the test_acc is:98.57
```

Finally, you will get:

```
epoch:9  train_acc:99.24 total_valid_acc:98.37 orig_test_acc:98.57 loss_total:1.6510787696788611 total_valid_max:98.37 in that epcoh, the orig_test_acc is:98.57
```

#### RoBerta_base

##### roberta base normal train

```
python Sentiment_task/roberta_base_normal_train.py --device 0 --epochs 10 --lr 1e-5 --batchsize 4 --warm_up_rate 0.1 --run_seed 4 --save_folder ./roberta_base_normal_train/ 
```

While training, the result will be shown on the screen and write into file "acc_out" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be:

```
epoch:0 train_acc:94.72759226713532 total_val_acc:92.65306122448979 orig_val_acc:93.46938775510205 total_test_acc:93.75 orig_test_acc:92.82786885245902
epoch:1 train_acc:97.01230228471002 total_val_acc:90.61224489795919 orig_val_acc:91.83673469387755 total_test_acc:91.70081967213115 orig_test_acc:91.18852459016394
epoch:2 train_acc:99.29701230228471 total_val_acc:89.79591836734694 orig_val_acc:91.42857142857143 total_test_acc:92.21311475409836 orig_test_acc:92.82786885245902
epoch:3 train_acc:99.7070884592853 total_val_acc:92.44897959183673 orig_val_acc:91.83673469387755 total_test_acc:93.54508196721312 orig_test_acc:93.64754098360656
epoch:4 train_acc:99.82425307557118 total_val_acc:89.79591836734694 orig_val_acc:91.0204081632653 total_test_acc:91.70081967213115 orig_test_acc:92.00819672131148
```

In this result, the "orig\_" means only the result of  real data, and the "total\_" means the result on both CF and real data.

Final result in "final_acc":

```
random seed:4orig_val_acc: 93.46938775510205orig_test_acc: 92.82786885245902
```

##### roberta base cf train

```
python Sentiment_task/roberta_base_cf_train.py --device 0 --epochs 10 --lr 1e-5 --batchsize 4 --warm_up_rate 0.1 --run_seed 4 --save_folder ./roberta_base_cf_train/ 
```

While training, the result will be shown on the screen and write into file "acc_out" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be:

```
epoch:0 train_acc:94.43468072642062 total_val_acc:90.81632653061224 orig_val_acc:89.79591836734694 total_test_acc:92.52049180327869 orig_test_acc:90.77868852459017
epoch:1 train_acc:96.74868189806678 total_val_acc:93.06122448979592 orig_val_acc:91.42857142857143 total_test_acc:93.54508196721312 orig_test_acc:90.98360655737704
epoch:2 train_acc:97.65670767428237 total_val_acc:91.22448979591837 orig_val_acc:89.79591836734694 total_test_acc:93.13524590163935 orig_test_acc:90.77868852459017
epoch:3 train_acc:98.7111892208553 total_val_acc:93.46938775510205 orig_val_acc:91.0204081632653 total_test_acc:94.4672131147541 orig_test_acc:92.41803278688525
epoch:4 train_acc:98.7990626830697 total_val_acc:93.06122448979592 orig_val_acc:91.0204081632653 total_test_acc:93.0327868852459 orig_test_acc:89.75409836065573
```

the final result is

```
random seed:4total_val_acc: 94.89795918367346orig_test_acc: 91.39344262295081
```

##### roberta base CRM

```
python Sentiment_task/roberta_base_CRM.py --device 0 --epochs 20 --lr 1e-3 --batchsize 4 --run_seed 4 --save_folder ./roberta_base_CRM/ --cf_model_folder ./roberta_base_cf_train/
```

While training, the result will be shown on the screen and write into file "cf_inference_out.log" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be:

```
epoch:0 train_acc:99.41417691857059 val_acc:98.36734693877551 test_acc:97.74590163934427
epoch:1 train_acc:99.53134153485648 val_acc:98.36734693877551 test_acc:97.54098360655738
epoch:2 train_acc:99.53134153485648 val_acc:97.55102040816327 test_acc:97.54098360655738
epoch:3 train_acc:99.53134153485648 val_acc:97.95918367346938 test_acc:97.74590163934427
epoch:4 train_acc:99.58992384299941 val_acc:97.95918367346938 test_acc:97.74590163934427
```

the final result is:

```
ramdom seed:4 max_val_acc:98.36734693877551 test_acc:97.74590163934427
```

#### RoBerta Large

##### roberta large normal train

```
python Sentiment_task/roberta_large_normal_train.py --device 0 --epochs 10 --lr 1e-5 --batchsize 4 --warm_up_rate 0.1 --run_seed 4 --save_folder ./roberta_large_normal_train/
```

While training, the result will be shown on the screen and write into file "acc_out" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be

```
epoch:0 train_acc:90.56824838898653 total_val_acc:87.34693877551021 orig_val_acc:91.0204081632653 total_test_acc:87.70491803278688 orig_test_acc:88.31967213114754
epoch:1 train_acc:98.41827768014059 total_val_acc:91.42857142857143 orig_val_acc:93.46938775510205 total_test_acc:92.62295081967213 orig_test_acc:93.44262295081967
epoch:2 train_acc:99.53134153485648 total_val_acc:93.46938775510205 orig_val_acc:95.10204081632654 total_test_acc:93.64754098360656 orig_test_acc:94.26229508196721
epoch:3 train_acc:99.82425307557118 total_val_acc:94.08163265306122 orig_val_acc:94.6938775510204 total_test_acc:93.13524590163935 orig_test_acc:94.05737704918033
epoch:4 train_acc:99.94141769185705 total_val_acc:93.46938775510205 orig_val_acc:93.46938775510205 total_test_acc:94.36475409836065 orig_test_acc:93.64754098360656
```

the final result in "final\_acc" is:

```
random seed:4orig_val_acc: 94.6938775510204orig_test_acc: 94.05737704918033
```

##### roberta large cf train

```
python Sentiment_task/roberta_large_cf_train.py --device 0 --epochs 10 --lr 1e-5 --batchsize 4 --warm_up_rate 0.1 --run_seed 4 --save_folder ./roberta_large_cf_train_new_env/
```

While training, the result will be shown on the screen and write into file "acc_out" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be

```
epoch:0 train_acc:96.5436438195665 total_val_acc:94.48979591836735 orig_val_acc:93.06122448979592 total_test_acc:94.97950819672131 orig_test_acc:93.23770491803279
epoch:1 train_acc:97.80316344463972 total_val_acc:94.08163265306122 orig_val_acc:93.46938775510205 total_test_acc:95.38934426229508 orig_test_acc:92.82786885245902
epoch:2 train_acc:98.50615114235501 total_val_acc:95.51020408163265 orig_val_acc:93.87755102040816 total_test_acc:95.08196721311475 orig_test_acc:93.0327868852459
epoch:3 train_acc:98.88693614528412 total_val_acc:95.10204081632654 orig_val_acc:93.87755102040816 total_test_acc:95.59426229508196 orig_test_acc:93.85245901639344
epoch:4 train_acc:99.06268306971295 total_val_acc:93.46938775510205 orig_val_acc:91.0204081632653 total_test_acc:94.56967213114754 orig_test_acc:91.80327868852459
```

the final result is

```
random seed:4total_val_max: 95.51020408163265orig_test_acc: 93.0327868852459
```

##### roberta large CRM

```
python Sentiment_task/roberta_large_CRM.py --device 0 --epochs 20 --lr 1e-3 --batchsize 4 --run_seed 4 --save_folder ./roberta_large_CRM/ --cf_model_folder ./roberta_large_cf_train/
```

While training, the result will be shown on the screen and write into file "cf_inference_out.log" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be

```
epoch:0 train_acc:99.35559461042764 total_val_acc:97.55102040816327 total_test_acc:98.36065573770492
epoch:1 train_acc:99.53134153485648 total_val_acc:97.55102040816327 total_test_acc:98.56557377049181
epoch:2 train_acc:99.7070884592853 total_val_acc:97.95918367346938 total_test_acc:98.77049180327869
epoch:3 train_acc:99.76567076742823 total_val_acc:97.95918367346938 total_test_acc:98.97540983606558
epoch:4 train_acc:99.82425307557118 total_val_acc:97.95918367346938 total_test_acc:98.77049180327869
```

the final result is:

```
ramdom seed:4 max_val_acc:98.77551020408163 test_acc:98.36065573770492
```





The rest of codes are coming soon