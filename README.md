# MSc Artificial Intelligence Dissertation
This repository contains the code and reports for my MSc Research Project.
Here, I implement several deep learning-based segmentation techniques and develop an attention-based iterative U-Net called the Attention Iternet.

## Deep-Vessel-Segmentation

### Abstract
Blood vessel segmentation is a crutial challenge which assists directly in several clinical fields. Many approaches exist from manual to computer-aided automatic segmentation, usually based on some variant of the U-Net. The invention of attention gates has taken the deep learning community by storm, particularly in semantic segmentation tasks with many state-of-the-art incorporating some form of attention gates. Attention is essentially a mechanism within a network which weights features by relative importance to a problem and then uses these features to solve the problem. Soft-attention is used in this project, where the weights of importance are learned through the standard backpropagation algorithm. This project explores vessel segmentation techniques using deep learning through the U-Net and builds on the current state-of-the-art known as the Iternet, to incorporate attention gates into its base module. Pre-processing included CLAHE, and green-channel extraction. A patch-based approach to training was incorporated. Experiments were conducted on the varying publicly available retinal blood vessel datasets; DRIVE, STARE, CHASE_DB1, and HRF which show the significant impact these attention gates can have on the performance of models, specifically in terms of sensitivity. Furthermore, attempts were made to use the trained models on different datasets of vessels from chloroplast in Bienertia chlorenchyma cells.


### References
This work uses multiple different segmentation repositories directly and indirectly. They can be found at (in no particular order):
- https://github.com/LeeJunHyun/Image_Segmentation
- https://github.com/HzFu/AGNet
- https://github.com/amri369/Pytorch-Iternet
- https://github.com/milesial/Pytorch-UNet
- https://github.com/MrGiovanni/UNetPlusPlus
- https://github.com/conscienceli/IterNet
- https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets
- https://github.com/orobix/retina-unet
