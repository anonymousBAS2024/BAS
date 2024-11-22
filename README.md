# Bridging Training-Sampling Gap in Conditional Diffusion Models: <br> Dynamic Interpolation and Self-Generated Augmentation for Image Restoration


## Environment Requirements

 We run the code on a computer with `RTX-4090`, and `24G` memory. The code was tested with `python 3.9.0`, `pytorch 2.4.0`, `cudatoolkit 12.1.0`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```
# create a virtual environment
conda create --name BAS python=3.9.0

# activate environment
conda activate BAS

# install pytorch & cudatoolkit
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## How to train
We train the model via running:

```
cd Desktop/codes/config/{task}
<br>
python train_ours.py -opt=options/train/ours.yml
```
## How to test
```
cd Desktop/codes/config/{task}
python test_ours.py -opt=options/test/ours.yml
```


The code of **BAS** is developed based on the code of [improved diffusion] (https://github.com/openai/improved-diffusion) and [IR-SDE] (https://github.com/Algolzw/image-restoration-sde)


