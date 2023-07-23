# Quantitative Portfolio Tool

The idea behind this project is to create a self-sustaining framework for managing and iterating on new and existing quant strategies that will make up our portfolio.

## Team

1. Anthony Givans

> Rising Junior, studying Math and CS with a minor in Astrophysics. Does Computer Vision Research and is really interested in applying Neural Nets to modern day problems. Currently working on making a Math and Physics LLM that takes a Math (or Physics) question and provides a step-by-step appraoch to solving said question. Also worked at Google (Summer 2023)

## How to Contribute

Note: these instructions assume a working knowledge of `git` and `github` and that you also have `git` already installed on your machine

* Firstly, `fork` this repository

> Look at the top right-hand corner of the page. You should see the option to `fork` the repo

* You will then clone the forked repo on your machine. You should see a green `code` button on the (`forked`) repo's page. Click that button, then copy the `.git` link that is present. Next, find a folder on your local machine where you are comfortable storing this code. I recommend `Desktop` but if you hae an existing file structure on your machine, feel free to use it. Once you have decided on where you want to store the code, enter the following command in your command prompt (eg, cmd prompt, bash, git bash, ubuntu, etc):

> git clone "the link you copied earlier that ended in `.git`"

At this point, you should have seen some output in your terminal and some files should have populated your directory. That's great!

* Next, we have some housekeeping stuff to go over:

We are using poetry to track our dependencies, which will make all our lives easier in the long-run. We already took care of this, so run this command in your terminal:

For Linux, macOs, or WSL do the following:

> pip install curl -sSl <https://install.python-poetry.org>

If you are using Windows Powershell, follow this [link](https://python-poetry.org/docs/#installing-with-the-official-installer) to get the instructions

Next, run

> poetry install

This command should have created a new environment and installed all the dependencies that you need to run the project in said environment. This command uses both the `pyproject.toml` and `poetry.lock` to setup the environment. In order to enter that environment, run

> poetry shell

Here, you should be able to run the files and test the code. We have a bit more housekeeping before we are done. Let's setup. Run the following:

> poetry add pre-commit

You may have noticed that we used `poetry` to install our dependency instead of `pip`. This is because `poetry` manages our dependencies and ensures that these dependencies are compatible. Also, `poetry` writes these chnages to the `poetry.lock` file so that the other persons working on the project will be able to download these dependencies as well. Whenever you have a new dependency, use `poetry add dependency` instead of `pip install dependency`. Lastly, run

> pre-commit install

There is a `.precommit-config.yaml` that specifies some interesting tasks that must be done (automatically) before commiting your code. These ensure that your code is bug free and conforms to the PEP8 standard. It also gives you a `coverage` summary, which shows you how much of your code is tested. Since this is early days, we are aiming for a **\>90%** coverage percentage.

As a quick example, run the following command

> poetry run tox

or, simply

> tox

This should run all the tests in the `/tests` folder and also provide and `coverage` summary for each run.

Now, let's talk about how you submit your code. The main branch is locked, meaning you can't directly make changes to it. You will have to create a new branch, using `git`, with the following command

> git checkout -b "name of branch" (without the quotes)

This will create a new branch, then switch to that branch, which is great!

After this, commit your changes to the branch and then submit your **PR** (Pull Request) for review.

### GOOD LUCK! :)
