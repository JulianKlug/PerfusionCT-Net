# PerfusionCT-Net
A UNet for the analysis of perfusion CT imaging in the setting of acute ischemic stroke. 


## Installation
`pip install -r requirements.txt`

##### Compatibility

- Environment must use python 3.7 (for torch and CUDA compatibility)

## References

- This is a fork of *ozan-oktay/Attention-Gated-Networks*   
- "Attention U-Net: Learning Where to Look for the Pancreas", MIDL'18, Amsterdam, [original paper](https://openreview.net/pdf?id=Skft7cijM) <br />
- Clinical implications: https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2762679

## Possibilites for further enhancement

- Downscaling to a 2.5 dimensional unet (eg. https://github.com/xyzacademic/multipathbmp), "A multi-path 2.5 dimensional convolutional neural network system for segmenting stroke lesions in brain MRI images", Xue et al, [paper](https://www.sciencedirect.com/science/article/pii/S2213158219304656)
- Multi-scale attention network: https://github.com/sinAshish/Multi-Scale-Attention
