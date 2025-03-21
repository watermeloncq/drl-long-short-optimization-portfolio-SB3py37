# Paper Code: **Deep Reinforcement Learning for Long-Short Portfolio Optimization**

## 1. About

### Paper URL：

https://arxiv.org/abs/2012.13773

In addition to the implementation code presented in the paper, this project provides supplementary code with the following algorithmic implementations for testing and learning purposes:

- PPO+ViT
- PPO+Resnet

### Training Rewards：

![training_rewards](https://github.com/user-attachments/assets/7d2ca16b-e727-4cda-a50f-fc5151786adc)


### Results：

![OPT_compare1](https://github.com/user-attachments/assets/f4c08dcb-a94f-4ebd-b3f8-4d94c066ac6b)

![OPT_compare2](https://github.com/user-attachments/assets/77e6f63f-ccc7-49d6-aaff-f214282d93c9)


This trading environment is developed based on [wassname](https://github.com/wassname)'s [rl-portfolio-management implementation](https://github.com/wassname/rl-portfolio-management), with improvements and migration to the Stable-Baselines3 (SB3) reinforcement learning framework. (Note: This code is compatible with SB3 under Python 3.7)



## 2. Required Python Packages

This code is only compatible with Python 3.7 and can be run on Linux or Windows 10/11 operating systems.

Execute the following commands to install the required Python packages:

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install stable-baselines3[extra]==1.3.0
pip install notebook
pip install einops
pip install tables
pip install seaborn
pip install tqdm
pip install openpyxl
```



## 3. Code Execution Steps and Important Notes

### (1) Code Execution Steps

After environment configuration and package installation, clone or download the repository locally to execute the Jupyter notebook files.

Step 1: Process data files by executing "./data/0. load chinese data 1d multindex.ipynb" .

Step 2: Execute the jupyter notebook files ("XX.ipynb") in the project root directory, for example:

- 10+1assets-drl-portfolio-arbitrage-stableBaseline3-PPO-VGG1-Tanh-AVGSharpe.ipynb : (This notebook implements the paper's methodology.)

<u>The training process often requires multiple complete restarts to achieve satisfactory backtesting performance, even with confirmed convergence of training rewards in each attempt. A single successful outcome may necessitate numerous training attempts from scratch.</u>

### (2) Important Notes（English & Chinese）

In "./data/0. load chinese data 1d multindex.ipynb" , the parameter test_split=0.08 defines the train-test split ratio. 

Critical Note: To prevent runtime errors, ensure that the test period length exceeds the random sampling interval used in training. For instance, when each training episode samples 128 trading days, the test period must contain a minimum of 128 days of data.

（在"./data/0. load chinese data 1d multindex.ipynb"  中，test_split=0.08 这个参数用于划分训练集和测试集。

注意：测试集的区间长度不能小于训练时随机采样的区间长度，否则会报错。例如，如果每个 episode 随机采样 128 个交易日的数据进行训练，那么测试集的区间长度就不能少于 128 天。）

### (3) How to use TensorBoard to Monitor Training Progress

Once the model training has commenced, navigate to the "./runs" directory and open a terminal. Execute the following command to monitor the training process:

```
tensorboard --logdir PPO-vgg1-Tanh-maxSharpe-arbitrage
```

 "PPO-vgg1-Tanh-maxSharpe-arbitrage" refers to the target folder within the runs directory.

## 4. Citation

If you use this code, please cite our paper:

```bibtex
@ARTICLE{2020arXiv201213773H,
       author = {{Huang}, Gang and {Zhou}, Xiaohua and {Song}, Qingyang},
        title = "{Deep Reinforcement Learning for Long-Short Portfolio Optimization}",
      journal = {arXiv e-prints},
     keywords = {Quantitative Finance - Computational Finance, Computer Science - Machine Learning, Quantitative Finance - Portfolio Management},
         year = 2020,
        month = dec,
          eid = {arXiv:2012.13773},
        pages = {arXiv:2012.13773},
          doi = {10.48550/arXiv.2012.13773},
archivePrefix = {arXiv},
       eprint = {2012.13773},
 primaryClass = {q-fin.CP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv201213773H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## 5.Comparative Analysis Code: Optimization Performance of DRL versus Traditional Optimization Models

URL: https://github.com/watermeloncq/OPT_comparison_for_paper
