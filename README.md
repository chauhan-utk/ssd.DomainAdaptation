# ssd.DomainAdaptation


## Basic git commands:
- ```git fetch``` : get remote commits to local repo
- ```git pull <repo> <branch-name>``` e.g. ```git pull origin master``` : get the commits from remote branch to current branch
- ```git branch -a``` : list all branches including remote branches
  - ```git branch -b <new-branch>```
  - ```git checkout <branch-name>``` : switch to the branch
- ```git status``` : list of files tracked and not tracked
- ```git add .``` **or** ```git add <file-names>``` : add files to staging area
  - ```git add "*.py"``` : add all python files to staging area

Undo mistakes using ```git revert```
There are several ways to undo a mistake, each with a different outcome. The best one is to use ```git revert```.
Check this [link](https://stackoverflow.com/questions/4114095/how-to-revert-git-repository-to-a-previous-commit) and [this](https://www.atlassian.com/git/tutorials/undoing-changes) tutorial.
- ```git revert HEAD``` - revert the immediate last commit by making exact opposite commit.
- ```git revert HEAD~2``` - go back two commits.
- ```git revert <sha1-1>..<sha1-3>``` - go back to the previous commit ranges.

Another tutorial [link](https://github.com/blog/2019-how-to-undo-almost-anything-with-git) by GitHub to undo stuff using ```git```.


## File list and their intentions:
- commonData.py: dataloader for common data

Create seperate branches for different approaches after the dataloader file is made and tested.
