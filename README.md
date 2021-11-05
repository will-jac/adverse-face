# Repo for Computer Security course project

## Installation

Note: I have a virtualenv set up for this, using a recent version of python (3.9.7). You can use this instead of installing everything yourself.

Assuming you're using pyenv:
`pyenv shell adverse-face`

Required libraries:

- deepface 
- numpy
- tensorflow 
- torch
- torchvision
- sklearn

Install all with one line:
`pip install numpy tensorflow torch torchvision deepface sklearn`

Required datasets:

- Labeled Faces in the Wild (LFW)

This is downloaded by torch into the data directory of this project upon running - see test_data_loader.ipynb

## Develoment

0. Clone the repository: `git clone https://github.com/will-jac/adverse-face`
1. Create a branch: `git branch image-distance`
2. Check out the branch: `git checkout image-distance`
3. Make changes, then add and commit them: `git add file; git commit -m "useful message"`
4. When you're ready, push to github: `git push`
5. When you're done making changes, merge them into main:
```{bash}
git checkout main
git merge main
```
This may require manual review. Always check that it is working before pushing, and always push everything to remote before merging.

Notes:

`main.py`

- Right now, the main.py file is just training & attacking with the no-box attack. This can be changed.
- Likely what will be best is to have a python file in the base dir of the project for generating each attack, and one for evaling each attack.
- TODO: Do this after eval.py is created? Or control the two with params passed into main.py?

`data/datasets/load_data.py`
- loads the data. `batch_size` is the number of images. 
- `batch_by_people` is a flag for no-box attacks - if True, each batch will contain `batch_size/2` images of one person, and `batch_size/2` of another person
- `shuffle` will shuffle the dataset. If False, each batch will be deterministic


## Attacks

### No-Box

Idea: train a surrogate auto-encoder model on unsupervised image reconstruction. Then, attack the surrogate model. Any attack that works well on the surrogate will probably work well on an actual FR system.

Code: see attacks/gan/no_box.py

Surrogate model: Currently using ResNet, but should probably switch to VGG Face. Just haven't had the time to code it up, as the only one released by the paper was ResNet.
Saved under attacks/gan/trained_ae

Attack Images: Saved under attacks/gan/trained_ae

### Obfuscated Gradients

TODO (Jack)

## Evaluated FR systems

TODO

## Plans

[x] Get datasets (using [LFW](http://vis-www.cs.umass.edu/lfw/))
[] Create and train working FR model 
[-] Create working attack
[x] Working no-box attack with ResNet surrogate model
[] Code a VGG surrogate model 
[] Obfuscated Gradient attacks
[] ???
[] Profit
