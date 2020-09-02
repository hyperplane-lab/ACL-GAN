### Paper

Yihao Zhao, Ruihai Wu, Hao Dong, "Unpaired Image-to-Image Translation using Adversarial Consistency Loss", ECCV 2020

arXiv: https://arxiv.org/abs/2003.04858 

### Code Base

[Xun Huang](http://www.cs.cornell.edu/~xhuang/), [Ming-Yu Liu](http://mingyuliu.net/), [Serge Belongie](https://vision.cornell.edu/se3/people/serge-belongie/), [Jan Kautz](http://jankautz.com/), "[Multimodal Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1804.04732)", ECCV 2018

### Code usage

For environment: conda env create -f acl-gan.yaml

For training: python train.py --config configs/male2female.yaml

For test: python test.py --config configs/male2female.yaml --input inputs/test_male.jpg --checkpoint ./models/test.pth 

### Experimental Results

##### Ablation study

<img src="figures/ablation_study.png" alt="ablation_study" style="zoom:50%;" />

##### Male-to-female

<img src="figures/male2female.png" alt="male2female" style="zoom:50%;" />

##### Glasses Removal

<img src="figures/glasses_removal.png" alt="glasses_removal" style="zoom:50%;" />

##### Selfie-to-anime

<img src="figures/selfie2anime.png" alt="selfie2anime" style="zoom:50%;" />



For more results, please refer to our paper.