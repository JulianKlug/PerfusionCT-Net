# PerfusionCT-Net
A UNet for the analysis of perfusion CT imaging in the setting of acute ischemic stroke. 

Please cite as: _Klug, J. et al. Bayesian Skip Net: Building on Prior Information for the Prediction and Segmentation of Stroke Lesions. in Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (eds. Crimi, A. & Bakas, S.) 168–180 (Springer International Publishing, 2021). doi:10.1007/978-3-030-72084-1_16._

Further work: [BayesianSkipNet](https://github.com/JulianKlug/BayesianSkipNet) 



## Installation
`pip install -r requirements.txt`

##### Compatibility

- Environment must use python 3.7 (for torch and CUDA compatibility)

## Getting started

- The main file for training can be found under `train_segmentation.py`. It takes a config file as argument, examples can be found in the `./config`folder. 
- A visdom server can launched as well for visualisation: `python -m visdom.server`

## References

- This is a fork of [*ozan-oktay/Attention-Gated-Networks*](https://github.com/ozan-oktay/Attention-Gated-Networks)
- "Attention U-Net: Learning Where to Look for the Pancreas", MIDL'18, Amsterdam, [original paper](https://openreview.net/pdf?id=Skft7cijM) <br />
- Clinical implications: https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2762679

## Possibilites for further enhancement

- Downscaling to a 2.5 dimensional unet (eg. https://github.com/xyzacademic/multipathbmp), "A multi-path 2.5 dimensional convolutional neural network system for segmenting stroke lesions in brain MRI images", Xue et al, [paper](https://www.sciencedirect.com/science/article/pii/S2213158219304656)
- Multi-scale attention network: https://github.com/sinAshish/Multi-Scale-Attention
