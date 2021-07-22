# SFace-Jittor
This is the code of SFace based on `Jittor`.

Paper: [《SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition》](https://ieeexplore.ieee.org/document/9318547)

## Abstract 
Deep face recognition has achieved great success due to large-scale training databases and rapidly developing loss functions. The existing algorithms devote to realizing an ideal idea: minimizing the intra-class distance and maximizing the inter-class distance. However, they may neglect that there are also low quality training images which should not be optimized in this strict way. Considering the imperfection of training databases, we propose that intra-class and inter-class objectives can be optimized in a moderate way to mitigate overfitting problem, and further propose a novel loss function, named sigmoid-constrained hypersphere loss (SFace). Specifically, SFace imposes intra-class and inter-class constraints on a hypersphere manifold, which are controlled by two sigmoid gradient re-scale functions respectively. The sigmoid curves precisely re-scale the intra-class and inter-class gradients so that training samples can be optimized to some degree. Therefore, SFace can make a better balance between decreasing the intra-class distances for clean examples and preventing overfitting to the label noise, and contributes more robust deep face recognition models. Extensive experiments of models trained on CASIA-WebFace, VGGFace2, and MS-Celeb-1M databases, and evaluated on several face recognition benchmarks, such as LFW, MegaFace and IJB-C databases, have demonstrated the superiority of SFace.

## Usage Instructions
1. Install Jittor with GPU support (Python 3.7).

2. Download the code.

3. The training datasets, CASIA-WebFace, VGGFace2 and MS1MV2, evaluation datasets can be downloaded from Data Zoo of [InsightFace](https://github.com/deepinsight/insightface). Then convert the training datasets into jpg format from the MXNet binary format. An example data structure for CASIA-WebFace is shown in `data/faces_webface_112x112/imgs.txt`.

## Train
Run the code to train a model.

(1) Train ResNet50, CASIA-WebFace, SFace.

- *With a single GPU*
```
CUDA_VISIBLE_DEVICES="0" python3 -u train_SFace_jittor.py --workers_id 0 --batch_size 256 --lr 0.1 --stages 50,70,80 --data_mode casia --net IR_50 --outdir ./results/IR_50-sface-casia --param_a 0.87 --param_b 1.2
```
- *With multiple GPUs*
```
CUDA_VISIBLE_DEVICES="0,1" mpirun -np 2 python3 -u train_SFace_jittor.py --workers_id 0,1 --batch_size 256 --lr 0.1 --stages 50,70,80 --data_mode casia --net IR_50 --outdir ./results/IR_50-sface-casia --param_a 0.87 --param_b 1.2
```

(2) Train MobileNet, CASIA-WebFace, SFace.

- *With a single GPU*
```
CUDA_VISIBLE_DEVICES="0" python3 -u train_SFace_jittor.py --workers_id 0 --batch_size 256 --lr 0.1 --stages 50,70,80 --data_mode casia --net MobileFaceNet --outdir ./results/Mobile-sface-casia --param_a 0.87 --param_b 1.2
```
- *With multiple GPUs*
```
CUDA_VISIBLE_DEVICES="0,1" mpirun -np 2 python3 -u train_SFace_jittor.py --workers_id 0,1 --batch_size 256 --lr 0.1 --stages 50,70,80 --data_mode casia --net MobileFaceNet --outdir ./results/Mobile-sface-casia --param_a 0.87 --param_b 1.2
```
