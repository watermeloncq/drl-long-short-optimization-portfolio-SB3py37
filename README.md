# Paper Code: **Deep Reinforcement Learning for Long-Short Portfolio Optimization**

## 1. About

This repository contains the implementation code for the revised version of the paper "Deep Reinforcement Learning for Long-Short Portfolio Optimization". This revised version was prepared in response to journal peer review and significantly expands upon the original work.

### Published Paper URL：

https://link.springer.com/article/10.1007/s10614-025-11143-4

**Original Preprint:** https://arxiv.org/abs/2012.13773

Note: The code in this repository corresponds to the published version in *Computational Economics* (2025). Key enhancements from the original preprint include robustness testing across six different random seeds and expanded cross-market validation. The dataset has been updated from the randomly selected CSI 300 constituents to the CSI 500 and S&P 500 test portfolios.




In addition to the implementation code presented in the paper, this project provides supplementary code with the following algorithmic implementations for testing and learning purposes:

- PPO+ViT
- PPO+Resnet



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

- 10+1CSI500-drl-portfolio-arbitrage-stableBaseline3-PPO-VGG1-Tanh-AVGSharpe-seed2.ipynb 
- 10+1S&P500-drl-portfolio-arbitrage-stableBaseline3-PPO-VGG1-Tanh-AVGSharpe-seed3.ipynb

<u>~~The training process often requires multiple complete restarts to achieve satisfactory backtesting performance, even with confirmed convergence of training rewards in each attempt. A single successful outcome may necessitate numerous training attempts from scratch.~~</u>

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

If you use this code, please cite our paper. The code in this repository corresponds to the **revised and extended version** of our work, which is currently under peer review at a journal. Until the final version is published, we recommend citing the original preprint available on arXiv:

```bibtex
@article{huangDeepReinforcementLearning2025a,
  title = {Deep Reinforcement Learning for Long-Short Portfolio Optimization},
  author = {Huang, Gang and Zhou, Xiaohua and Song, Qingyang},
  year = {2025},
  month = oct,
  journal = {Computational Economics},
  issn = {1572-9974},
  doi = {10.1007/s10614-025-11143-4},
  urldate = {2025-10-15},
  langid = {english}
}
```

## 5.Comparative Analysis Code: Optimization Performance of DRL versus Traditional Optimization Models

URL: https://github.com/watermeloncq/OPT_comparison_for_paper
