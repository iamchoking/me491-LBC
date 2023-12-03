# Starting point for ME491-20190673

this is a private homework repo for Learning based control.

Basic manipulation references this [gist](https://gist.github.com/injoonH/84f05d64b847cc18b9aeb597362fb512) by injoonH:

Cloning repo (+with upstream)
```sh
git clone git@github.com:iamchoking/me491-LBC
git rmote add upstream git@github.com:railabatkaist/2023_ME491.git
```
Manipulating Repo
```sh
# Initialize with Upstream
git clone --origin upstream git@github.com:railabatkaist/2023_ME491.git me491
cd me491
# (create a new (private) <repo> in your github)
git remote add origin git@github.com:<github-id>/<repo>.git
git checkout main
git push -u origin main

# get a branch from upstream
git fetch upstream
git checkout <branch>
git push -u origin <branch>

# sync an (existing) branch from upstream
git checkout <branch>
# (do NOT do "origin/<branch>")
git merge upstream/<branch>
git push -u origin <branch>
# may need to add options like -f

# if you have your "homework" branch that needs to incorporate this as a rabase, do:
git checkout <branch2>
git rebase <branch>
git push -u origin <branch2>

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
