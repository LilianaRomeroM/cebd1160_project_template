# cebd1160_project_template
Instructions and template for final projects.

| Name | Date |
|:-------|:---------------|
|Liliana Romero Molina | Completion date|

-----

### Resources
Your repository should include the following:

- Python script for your analysis
- Results figure/saved file
- Dockerfile for your experiment
- runtime-instructions in a file named RUNME.md

-----

## Research Question

The objective of this exercise is to determine whether, based on the information of the dataset (physiological characteristics and blood serum measurements), it is possible to predict the diabetes progression one year after a baseline measurement.

### Abstract

We have a dataset containing 442 samples from patients with diabetes, with 10 measurements established one year after the baseline, along with their respective qualitative assessment of disease progression.

Using this information as a base, we could identify if these factors have relevance in the progression of the disease (if they are predictors), and from there establish mitigation methods on them in an early way.

As a result of the analyses carried out (Lasso method, with add. Ridge) I find that the model as a whole does not accurately predict the progression of the disease, however it indicates that we should focus our attention on a fundamental factor (body mass index).  
It is also likely that by using a more complex (non-linear) method, a better prediction will be obtained. 

### Introduction

Diabetes dataset

Ten baseline variables, age, sex, body mass index, average blood pressure, and six blood serum measurements were obtained for each of n = 442 diabetes patients, as well as the response of interest, a quantitative measure of disease progression one year after baseline. 
 
Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times n_samples(i.e. the sum of squares of each column totals 1). 

https://scikit-learn.org/stable/datasets/index.html#diabetes-dataset

According to an additional source, the real 10 variables names are (for the exercise we are keeping the original file descriptions):

age sex bmi map tc ldl hdl tch ltg glu y

https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

According to the initial revision, the information is complete, without null or missing data.   In the same way, I do not see "atypical" data.  It is understood that as a result of these steps the data is ready to start exploration and modeling.

### Methods


Brief (no more than 1-2 paragraph) description about how you decided to approach solving it. Include:

- pseudocode for this method (either created by you or cited from somewhere else)

As drivers for defining the method, I took into account a visualization input (Heatmap) and the p-value of each of the independent variables obtained from an initial linear regression.  
According to both observations, apparently only a handful of variables have a significant influence on the target variable, in this case the progression of diabetes.  
Therefore, I opted to apply the Lasso model, understanding that it performs "eliminations" of non-significant variables within its process.

Translated with www.DeepL.com/Translator

![alt text](https://github.com/LilianaRomeroM/cebd1160_project_template/blob/master/diabetes7plots/heatseaborn.png)


    Coefficient  Std Error  t value  p-value
0      152.1335      2.576   59.061    0.000
1      -10.0122     59.749   -0.168    0.867
2     -239.8191     61.222   -3.917    0.000
3      519.8398     66.534    7.813    0.000
4      324.3904     65.422    4.958    0.000
5     -792.1842    416.684   -1.901    0.058
6      476.7458    339.035    1.406    0.160
7      101.0446    212.533    0.475    0.635
8      177.0642    161.476    1.097    0.273
9      751.2793    171.902    4.370    0.000
10      67.6254     65.984    1.025    0.306

Variables with p-values < 0.05 (relationship to the response variable is signifcant): x2(sex), x3(bmi), x4(bp), x9(ltg)




### Results

Brief (2 paragraph) description about your results. Include:

- At least 1 figure
- At least 1 "value" that summarizes either your data or the "performance" of your method
- A short explanation of both of the above

### Discussion
Brief (no more than 1-2 paragraph) description about what you did. Include:

- interpretation of whether your method "solved" the problem
- suggested next step that could make it better.

### References
All of the links

-------
