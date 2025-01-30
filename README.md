# **Vertical Federated Learning with MNIST using PyTorch**  

This repository implements a **Vertical Federated Learning (VFL)** approach on the **MNIST dataset** using **PyTorch**. In **VFL**, different clients hold **different feature subsets** of the same dataset. This implementation **splits the MNIST images vertically** into two parts, trains separate neural networks on each half, and then combines their predictions for evaluation.  

## **How It Works**  

### **Neural Network for Each Client**  
- Each client gets **half of the image features** (left or right half).  
- A simple **fully connected neural network (FCN)** processes each half separately.  

### **Training Process**  
- Each client trains a local model using **Stochastic Gradient Descent (SGD)**.  
- **Federated Averaging (FedAvg)** is used to aggregate model weights from both clients.  

### **Evaluation**  
- The two trained models are combined by **summing their outputs**.  
- The final classification is done based on the **averaged predictions**.  

## **Installation & Setup**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/phantomkali/Federated-learning.git
cd Federated-learning
```  

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```  

### **3. Run the Training & Evaluation**  
```bash
python train.py
```  

## **Dependencies**  
- **Python 3.8+**  
- **PyTorch**  
- **torchvision**  
- **numpy**  

## **Files**  
- **train.py**: Main script to train and evaluate the federated learning models.  

## **Results**  

| Method  | Accuracy |
|---------|----------|
| **FedAvg** | **87.47%** |
| **FedProx (Client 1)** | **49.30%** |
| **FedProx (Client 2)** | **50.70%** |

## **Future Improvements**  
✅ Use **Secure Aggregation** to ensure privacy when averaging weights.  
✅ Improve **model architecture** for better feature extraction.  
✅ Implement **communication protocols** for real-world federated learning scenarios.  
