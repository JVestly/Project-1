Project 1 in FYS-STK4155

Project authors: Amund Solum Alsvik og Jørgen Vestly

In this project, we're investigating different methods for performing regression.
We will mainly compare results for using different methods such as OLS, Lasso, and Ridge, using both analytical and numerical methods.
Moreover, we will discuss the important role of bias-variance tradeoff for our models.

The code folder contains all the scripts for our project.
functions.py - defines all functions needed for the computations in our project.
classes.py - defines the gradient descent and resampling classes.
plots.py - defines the different plotting methods used for presenting results.
imports.py - contains all needed imports for the project.
main.py - main file, from where all the code is ran.

In order for the code to work, there is a list of required installations in the main folder "Project-1"
requirements.txt - a list of mandatory installations to run the code.

Run main.py to perform all testing.
NOTE:
We noticed that our last tests for convergence of the various gradient descent methods started leading to overflow issues, although the code remains unchanged. This could be due to an issue with our console. We are working on identifying the issue. Nevertheless, the functioning plot can be found in the report. 
