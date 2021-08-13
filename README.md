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

While training, the result will be shown on the screen and written into file "acc_list" under save_folder

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

While training, the result will be shown on the screen and written into file "acc_list" under save_folder

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

While training, the result will be shown on the screen and written into file "acc_list" under save_folder

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

While training, the result will be shown on the screen and written into file "acc_out" under save_folder, and the final result would be found in final_acc

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

While training, the result will be shown on the screen and written into file "acc_out" under save_folder, and the final result would be found in final_acc

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

While training, the result will be shown on the screen and written into file "cf_inference_out.log" under save_folder, and the final result would be found in final_acc

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

While training, the result will be shown on the screen and written into file "acc_out" under save_folder, and the final result would be found in final_acc

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

While training, the result will be shown on the screen and written into file "acc_out" under save_folder, and the final result would be found in final_acc

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

While training, the result will be shown on the screen and written into file "cf_inference_out.log" under save_folder, and the final result would be found in final_acc

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



##### roberta large CRM generator

We provided a trained generator in random seed 4, you can use this command to get result:

```
python Sentiment_task/roberta_large_CRM_gen.py --device 0 --epochs 20 --lr 1e-3 --batchsize 4 --run_seed 4 --save_folder ./roberta_large_CRM_gen_result/ --cf_model_folder ./roberta_large_cf_train/
```

While training, the result will be shown on the screen and written into file "cf_inference_out.log" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be

```
epoch:0 train_acc:98.65260691271236 val_acc:93.46938775510205 test_acc:94.67213114754098
epoch:1 train_acc:98.59402460456943 val_acc:93.46938775510205 test_acc:94.67213114754098
epoch:2 train_acc:98.53544229642648 val_acc:93.46938775510205 test_acc:94.67213114754098
epoch:3 train_acc:98.65260691271236 val_acc:93.06122448979592 test_acc:94.67213114754098
epoch:4 train_acc:98.76977152899825 val_acc:92.24489795918367 test_acc:94.05737704918033
```

the final result is:

```
ramdom seed:4 max_val_acc:93.46938775510205 test_acc:94.67213114754098
```



## NLI Task

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

##### Roberta base normal train

```
python NLI_task/roberta_base_normal_train.py --device 0 --epochs 10 --lr 1e-5 --batchsize 4 --warm_up_rate 0.1 --run_seed 4 --save_folder ./NLI_tasks/roberta_base_normal_train/
```

While training, the result will be shown on the screen and written into file "acc_out" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be:

```
epoch:0 train_acc:73.64945978391357 orig_val_acc:73.0 orig_test_acc:71.75
epoch:1 train_acc:91.41656662665066 orig_val_acc:81.5 orig_test_acc:84.75
epoch:2 train_acc:95.85834333733493 orig_val_acc:79.5 orig_test_acc:82.5
epoch:3 train_acc:98.49939975990397 orig_val_acc:79.0 orig_test_acc:82.75
epoch:4 train_acc:99.27971188475391 orig_val_acc:78.5 orig_test_acc:83.0
```

the final result in final_acc is:

```
random seed:4orig_val_acc: 81.5orig_test_acc: 84.75
```

##### Roberta base cf train

```
python NLI_task/roberta_base_cf_train.py --device 0 --epochs 10 --lr 1e-5 --batchsize 4 --warm_up_rate 0.1 --run_seed 4 --save_folder ./NLI_tasks/roberta_base_cf/
```

While training, the result will be shown on the screen and written into file "acc_out" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be:

```
epoch:0 train_acc:67.6470588235294 total_val_acc:64.5 orig_val_acc:58.5 total_test_acc:62.05 orig_test_acc:67.5
epoch:1 train_acc:84.47779111644658 total_val_acc:74.0 orig_val_acc:78.5 total_test_acc:73.05 orig_test_acc:77.5
epoch:2 train_acc:91.15246098439376 total_val_acc:76.6 orig_val_acc:79.5 total_test_acc:76.5 orig_test_acc:85.75
epoch:3 train_acc:94.38175270108043 total_val_acc:78.1 orig_val_acc:80.5 total_test_acc:78.45 orig_test_acc:85.75
epoch:4 train_acc:95.59423769507804 total_val_acc:76.8 orig_val_acc:80.5 total_test_acc:76.8 orig_test_acc:83.0
```

the final result in final_acc is:

```
random seed:4total_val_acc: 78.4orig_test_acc: 83.5
```

##### Roberta base EF

```
python NLI_task/roberta_base_EF.py --device 0 --epochs 20 --lr 1e-3 --batchsize 8 --run_seed 4 --save_folder ./NLI_tasks/roberta_base_EF/ --cf_model_folder ./NLI_tasks/roberta_base_cf/
```

While training, the result will be shown on the screen and written into file "cf_inference_out.log" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be

```
epoch:0 train_acc:97.05882352941177 val_acc:87.5 test_acc:88.5
epoch:1 train_acc:99.45978391356543 val_acc:91.5 test_acc:92.75
epoch:2 train_acc:99.87995198079231 val_acc:89.0 test_acc:92.75
epoch:3 train_acc:99.93997599039616 val_acc:90.5 test_acc:92.5
epoch:4 train_acc:100.0 val_acc:92.0 test_acc:92.5
```

the final result in final_acc is:

```
random seed:4max_val_acc: 94.0 final_test_acc: 93.25
```

##### Roberta base LF

```
python NLI_task/roberta_base_LF.py --device 0 --epochs 20 --lr 1e-3 --batchsize 8 --run_seed 4 --save_folder ./NLI_tasks/roberta_base_LF/ --cf_model_folder ./NLI_tasks/roberta_base_cf/
```

While training, the result will be shown on the screen and written into file "cf_inference_out.log" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be

```
epoch:0 train_acc:99.2797118847539 val_acc:90.5 test_acc:91.0
epoch:1 train_acc:99.75990396158463 val_acc:92.0 test_acc:93.0
epoch:2 train_acc:99.87995198079231 val_acc:93.0 test_acc:93.75
epoch:3 train_acc:99.93997599039616 val_acc:95.0 test_acc:93.75
epoch:4 train_acc:100.0 val_acc:96.0 test_acc:95.25
```

the final result in final_acc is:

```
random seed:4max_val_acc: 96.0 final_test_acc: 95.25
```

##### Roberta base MF

```
python NLI_task/roberta_base_MF.py --device 0 --epochs 20 --lr 1e-3 --batchsize 8 --run_seed 4 --save_folder ./NLI_tasks/roberta_base_MF/ --cf_model_folder ./NLI_tasks/roberta_base_cf/
```



##### Roberta large normal train

```
python NLI_task/roberta_large_normal_train.py --device 0 --epochs 10 --lr 1e-5 --batchsize 4 --warm_up_rate 0.1 --run_seed 4 --save_folder ./NLI_tasks/roberta_large_normal_train/
```

While training, the result will be shown on the screen and written into file "acc_out" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be:

```
epoch:0 train_acc:68.18727490996399 orig_val_acc:64.0 orig_test_acc:63.25
epoch:1 train_acc:92.4969987995198 orig_val_acc:90.5 orig_test_acc:86.25
epoch:2 train_acc:96.03841536614645 orig_val_acc:84.0 orig_test_acc:84.5
epoch:3 train_acc:98.5594237695078 orig_val_acc:86.0 orig_test_acc:88.5
epoch:4 train_acc:99.69987995198079 orig_val_acc:87.5 orig_test_acc:88.25
```

the final result in final_acc is:

```
random seed:4orig_val_acc: 90.5orig_test_acc: 86.25
```

##### Roberta large cf  train

```
python NLI_task/roberta_large_cf_train.py --device 0 --epochs 10 --lr 1e-5 --batchsize 4 --warm_up_rate 0.1 --run_seed 4 --save_folder ./NLI_tasks/roberta_large_cf/
```

While training, the result will be shown on the screen and written into file "acc_out" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be:

```
epoch:0 train_acc:73.01320528211285 total_val_acc:69.8 orig_val_acc:68.5 total_test_acc:66.55 orig_test_acc:68.75
epoch:1 train_acc:87.92316926770708 total_val_acc:78.6 orig_val_acc:84.0 total_test_acc:77.05 orig_test_acc:84.5
epoch:2 train_acc:92.59303721488595 total_val_acc:81.2 orig_val_acc:85.5 total_test_acc:81.05 orig_test_acc:86.75
epoch:3 train_acc:93.99759903961585 total_val_acc:81.9 orig_val_acc:83.5 total_test_acc:80.7 orig_test_acc:85.0
epoch:4 train_acc:95.73829531812726 total_val_acc:81.0 orig_val_acc:83.0 total_test_acc:80.15 orig_test_acc:84.5
```

the final result in final_acc is:

```
random seed:4total_val_acc: 81.9orig_test_acc: 85.0
```

##### Roberta large EF

```
python NLI_task/roberta_large_EF.py --device 0 --epochs 20 --lr 1e-3 --batchsize 8 --run_seed 4 --save_folder ./NLI_tasks/roberta_large_EF/ --cf_model_folder ./NLI_tasks/roberta_large_cf/
```

While training, the result will be shown on the screen and written into file "cf_inference_out.log" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be

```
epoch:0 train_acc:96.0984393757503 val_acc:91.5 test_acc:92.5
epoch:1 train_acc:98.73949579831933 val_acc:89.5 test_acc:94.25
epoch:2 train_acc:99.39975990396158 val_acc:92.5 test_acc:94.25
epoch:3 train_acc:99.39975990396158 val_acc:89.0 test_acc:94.75
epoch:4 train_acc:99.21968787515006 val_acc:91.5 test_acc:94.0
```

the final result in final_acc is:

```
random seed:4max_val_acc: 93.5 final_test_acc: 96.0
```

##### Roberta large LF

```
python NLI_task/roberta_large_LF.py --device 0 --epochs 20 --lr 1e-3 --batchsize 8 --run_seed 4 --save_folder ./NLI_tasks/roberta_large_LF/ --cf_model_folder ./NLI_tasks/roberta_large_cf/
```

While training, the result will be shown on the screen and written into file "cf_inference_out.log" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be

```
epoch:0 train_acc:97.47899159663865 val_acc:92.0 test_acc:94.5
epoch:1 train_acc:98.67947178871549 val_acc:89.0 test_acc:94.75
epoch:2 train_acc:99.2797118847539 val_acc:90.5 test_acc:96.5
epoch:3 train_acc:99.69987995198079 val_acc:94.5 test_acc:96.25
epoch:4 train_acc:99.69987995198079 val_acc:93.0 test_acc:96.0
```

the final result in final_acc is:

```
random seed:4max_val_acc: 94.5 final_test_acc: 96.25
```



##### Roberta large MF

```
python NLI_task/roberta_large_MF.py --device 0 --epochs 20 --lr 1e-3 --batchsize 8 --run_seed 4 --save_folder ./NLI_tasks/roberta_large_MF/ --cf_model_folder ./NLI_tasks/roberta_large_cf/
```

While training, the result will be shown on the screen and written into file "cf_inference_out.log" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be

```
epoch:0 train_acc:98.67947178871549 val_acc:91.5 test_acc:93.25
epoch:1 train_acc:99.21968787515006 val_acc:92.5 test_acc:93.0
epoch:2 train_acc:99.21968787515006 val_acc:89.5 test_acc:93.0
epoch:3 train_acc:99.2797118847539 val_acc:91.0 test_acc:92.25
epoch:4 train_acc:99.51980792316927 val_acc:89.5 test_acc:91.5
```

the final result in final_acc is:

```
random seed:4max_val_acc: 93.0 final_test_acc: 92.5
```



##### Roberta large nli normal train

```
nohup python NLI_task/roberta_large_nli_normal_train.py --device 0 --epochs 10 --lr 1e-5 --batchsize 4 --warm_up_rate 0.1 --run_seed 4 --save_folder ./NLI_tasks/roberta_large_nli_normal_train/ &
```

While training, the result will be shown on the screen and written into file "acc_out" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be:

```
epoch:0 train_acc:93.6374549819928 orig_val_acc:88.5 orig_test_acc:91.0
epoch:1 train_acc:97.35894357743098 orig_val_acc:92.0 orig_test_acc:88.75
epoch:2 train_acc:99.03961584633853 orig_val_acc:88.0 orig_test_acc:89.5
epoch:3 train_acc:99.75990396158464 orig_val_acc:89.5 orig_test_acc:91.25
epoch:4 train_acc:99.87995198079231 orig_val_acc:89.0 orig_test_acc:91.5
```

the final result in final_acc is:

```
random seed:4orig_val_acc: 92.0orig_test_acc: 88.75
```



##### Roberta large nli cf train

```
python NLI_task/roberta_large_nli_cf_train.py --device 0 --epochs 10 --lr 1e-5 --batchsize 4 --warm_up_rate 0.1 --run_seed 4 --save_folder ./NLI_tasks/roberta_large_nli_cf/
```

While training, the result will be shown on the screen and written into file "acc_out" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be:

```
epoch:0 train_acc:85.21008403361344 total_val_acc:79.1 orig_val_acc:79.5 total_test_acc:78.7 orig_test_acc:84.0
epoch:1 train_acc:90.73229291716687 total_val_acc:82.2 orig_val_acc:89.0 total_test_acc:81.25 orig_test_acc:89.25
epoch:2 train_acc:93.50540216086435 total_val_acc:81.5 orig_val_acc:88.5 total_test_acc:81.9 orig_test_acc:88.25
epoch:3 train_acc:95.22208883553421 total_val_acc:82.9 orig_val_acc:88.0 total_test_acc:82.45 orig_test_acc:88.0
epoch:4 train_acc:96.24249699879952 total_val_acc:83.0 orig_val_acc:86.5 total_test_acc:82.6 orig_test_acc:87.75
```

the final result in final_acc is:

```
random seed:4total_val_acc: 83.9orig_test_acc: 87.5
```

##### Roberta large nli EF

```
python NLI_task/roberta_large_nli_EF.py --device 0 --epochs 20 --lr 1e-3 --batchsize 8 --run_seed 4 --save_folder ./NLI_tasks/roberta_large_nli_EF/ --cf_model_folder ./NLI_tasks/roberta_large_nli_cf/
```

While training, the result will be shown on the screen and written into file "cf_inference_out.log" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be

```
epoch:0 train_acc:98.9795918367347 val_acc:93.5 test_acc:91.75
epoch:1 train_acc:99.15966386554622 val_acc:92.5 test_acc:91.5
epoch:2 train_acc:99.75990396158464 val_acc:94.0 test_acc:93.25
epoch:3 train_acc:99.87995198079231 val_acc:93.5 test_acc:91.5
epoch:4 train_acc:100.0 val_acc:94.5 test_acc:94.75
```

the final result in final_acc is:

```
random seed:4max_val_acc: 94.5 final_test_acc: 94.75
```

#####  Roberta large nli LF

```
python NLI_task/roberta_large_nli_LF.py --device 0 --epochs 20 --lr 1e-3 --batchsize 8 --run_seed 4 --save_folder ./NLI_tasks/roberta_large_nli_LF/ --cf_model_folder ./NLI_tasks/roberta_large_nli_cf/
```

While training, the result will be shown on the screen and written into file "cf_inference_out.log" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be

```
epoch:0 train_acc:99.39975990396158 val_acc:93.0 test_acc:92.5
epoch:1 train_acc:99.57983193277312 val_acc:94.5 test_acc:92.5
epoch:2 train_acc:99.93997599039616 val_acc:93.5 test_acc:94.0
epoch:3 train_acc:99.93997599039616 val_acc:93.0 test_acc:94.0
epoch:4 train_acc:100.0 val_acc:94.5 test_acc:94.5
```

the final result in final_acc is:

```
random seed:4max_val_acc: 95.5 final_test_acc: 93.5
```



##### Roberta large nli MF

```
python NLI_task/roberta_large_nli_MF.py --device 0 --epochs 20 --lr 1e-3 --batchsize 8 --run_seed 4 --save_folder ./NLI_tasks/roberta_large_nli_MF/ --cf_model_folder ./NLI_tasks/roberta_large_nli_cf/
```

While training, the result will be shown on the screen and written into file "cf_inference_out.log" under save_folder, and the final result would be found in final_acc

If you have same environment as me, the first five results would be

```
epoch:0 train_acc:99.63985594237695 val_acc:92.0 test_acc:94.75
epoch:1 train_acc:99.81992797118848 val_acc:93.0 test_acc:94.5
epoch:2 train_acc:97.95918367346938 val_acc:92.0 test_acc:91.5
epoch:3 train_acc:99.63985594237695 val_acc:90.5 test_acc:90.0
epoch:4 train_acc:99.81992797118848 val_acc:94.5 test_acc:95.0
```

the final result in final_acc is:

```
random seed:4max_val_acc: 96.0 final_test_acc: 94.25
```



While exploring this issue, we found that for tasks that a  factual sample with multiple counterfactual samples, we could directly try to generate results after Early Fusion instead of generating different counterfactual samples. The generator, which differs from the standard CRM generator in its loss function (equation 6 in paper), expects the label of the counterfactual sample it generates to be close to all labels that are different from the real sample at the same time. 

This can be formulated as:
$$
\omega = \arg\min_\omega r(\boldsymbol{u_c^*,\overline{u}_c}) + \sum_{c\neq y} \gamma l(c, f(\boldsymbol{x^*_c-u_c^*|\hat{\theta}})) + r(\boldsymbol{u,\overline{u}}) +\gamma l(c, f(\boldsymbol{x-u|\hat{\theta}}))
$$

```
python NLI_task/roberta_large_CRM_nli_gen.py --device 0 --epochs 20 --lr 1e-3 --batchsize 4 --run_seed 4 --save_folder ./NLI_tasks/roberta_large_CRM_nli_gen_result/ --cf_model_folder ./NLI_tasks/roberta_large_cf_train/
```

If you have same environment as me, the first five results would be (you can find this in saving folder)

```
epoch:0 train_acc:93.51740696278512 val_acc:83.0 test_acc:84.75
epoch:1 train_acc:93.75750300120048 val_acc:84.0 test_acc:84.5
epoch:2 train_acc:93.6374549819928 val_acc:84.0 test_acc:84.75
epoch:3 train_acc:93.15726290516207 val_acc:84.0 test_acc:84.5
epoch:4 train_acc:93.937575030012 val_acc:85.0 test_acc:85.25
```

the final result is:

```
random seed:4test acc:86.75
```

