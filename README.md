# move to following HW branch

this is a private homework repo for Learning based control.

Basic manipulation references this [gist](https://gist.github.com/injoonH/84f05d64b847cc18b9aeb597362fb512) by injoonH:

```sh
# Clone
git clone --origin upstream git@github.com:railabatkaist/2023_ME491.git me491
cd me491
# (create a new (private) <repo> in your github)
git remote add origin git@github.com:<github-id>/<repo>.git
git checkout main
git push -u origin main

# sync a branch from upstream
git fetch upstream
git checkout <branch>
git push -u origin <branch>

# adding your homework branch
# (in web, create your homework branch from skeleton branch)
git fetch origin
git checkout <mybranch>
git push origin <mybranch>

# Check remotes
git remote -v

# (?) Get updates from the upstream
git fetch upstream
git merge upstream/main # Or other branches
```
