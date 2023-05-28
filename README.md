# Repository Setup

## 1. Clone the repo

```bash
git clone <repository_url>
```
## 2. Add 'Sample_For_Assignment.pdf' file to the local repository

## 3. Create a conda environment: 
```bash
conda create -n senzmateOCR python=3.9
```

## 4. Activate the conda environment:
```bash
conda activate senzmateOCR
```

## 5. Install the dependencies

### If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install
```bash
python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### If you have no available GPU on your machine, please run the following command to install the CPU version
```bash
python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### Install PaddleOCR Whl Package
```bash
pip install "paddleocr>=2.0.1" # Recommend to use version 2.0.1+
```
### Force install pymupdf<=1.19.0 to avoid conflicts
```bash
pip install "pymupdf==1.19.0" 

```
## 6. Run pdf_to_json.py file
```bash
python3 pdf_to_json.py
```
