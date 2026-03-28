# 模型训练方式
### 创建环境
```bash
conda create -n xxx python==3.8.0
```
### 安装依赖
```bash
pip install -r requirements.txt
```
### 安装1.9.1版本的torch
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
### 开始训练
```bash
sh run.sh
```