# ssd.DomainAdaptation

## GRL branch

One architecture is to train the network with a batch in which half the images are from source domain (with ground truth annotations) and half the images are from target domain (without ground truth annotations). The baseline will be:
- n/w trained with source and tested on target (lower bound)
- n/w trained with both source and target annotations (upper bound)


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
View git branches in terminal in nice graph form - ```git log --graph --decorate --oneline```


Check if ```git pull``` is needed: ```git remote -v update```
Add an empty directory to the repo (useful when want to store stuff that should not be sent to remote)
- create directory (e.g. ```.logs```)
- inside ```.logs``` create ```.gitignore``` and add following:
    ```Shell
    # Ignore everything in this directory
    *
    # Except this file
    !.gitignore
    ```

Another tutorial [link](https://github.com/blog/2019-how-to-undo-almost-anything-with-git) by GitHub to undo stuff using ```git```.

If want to overwrite local files with remote:
```Shell
git fetch --all
git reset --hard origin/master
git pull origin master
```


## File list and their intentions:
- commonData.py: dataloader for common data


vgg lower layers give value NaN initially. Don't know if this is good or bad. Check [this](https://github.com/tensorflow/tensorflow/issues/3199) and [this](https://stackoverflow.com/questions/33712178/tensorflow-nan-bug?newreg=c7e31a867765444280ba3ca50b657a07) link for more information on this NaN values issue.
