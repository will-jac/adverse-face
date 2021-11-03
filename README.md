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

## Plans

[] Get datasets (shell script?)
[] Create and train working FR model 
[] Create and train (if necessary) working attack(s)
[] ???
[] Profit
