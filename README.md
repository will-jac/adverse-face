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
- pillow (PIL)
- sklearn

Install all with one line:
`pip install numpy tensorflow torch torchvision deepface pillow sklearn`

Required datasets:

- Labeled Faces in the Wild (LFW)

This is downloaded by torch into the data directory of this project upon running - see test_data_loader.ipynb

## Usage

## Plans

[] Get datasets (shell script?)
[] Create and train working FR model 
[] Create and train (if necessary) working attack(s)
[] ???
[] Profit