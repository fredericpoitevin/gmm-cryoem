# GMM in cryoEM

## Quick installation instructions

1. Make sure you have conda installed on your computer, or [download miniconda here](https://docs.conda.io/en/latest/miniconda.html).

2. On GitHub, fork this repository in your personal account

3. Run the following commands:
```
git clone https://github.com/yourusername/gmm-cryoem.git gmm-cryoem
cd gmm-cryoem
conda env create --file environment.yml
conda activate gmm-cryoem
```

## Contributing

1. *Add Fred's master branch as remote* (only need to do this once):
```
git remote add fred https://github.com/fredericpoitevin/gmm-cryoem
```

2. *Update your fork regularly* (before you start working) by pulling from Fred's master branch:
```
git pull fred master
```

3. *Use feature branches*: if you want to edit existing or add new code, make a new branch and work there. Once you are done, make a pull request to Fred's master branch:
```
git checkout -B new-branch-name
... edit, add, etc. ...
git push origin new-branch-name
git checkout master
```

[GIT CHEAT SHEET](https://education.github.com/git-cheat-sheet-education.pdf)
