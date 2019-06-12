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

Using the information on the progression of diabetes in patients after one year of diagnosis, can we determine and predict for a new patient diagnosed with this disease, the main factor(s) to control in order to minimize such progression? 

### Abstract

4 sentence longer explanation about your research question. Include:

Derived from Diffusion-Weighted Magnetic Resonance Imaging (DWI, d-MRI), we have derived "maps" of structural connectivity between brain regions.
Using these data, we may be able to understand relationships between brain regions and their relative connectivity, which can then be used for targetted interventions in neurodegenerative diseases.
Here, we tried to predict the connectivity between two unique brain regions based on all other known brain connectivity maps.
Based on the preliminary performance of this regressor, we found that the current model didn't provide consistent performance, but shows promise for success with more sophisticated methods.


- opportunity (what data do we have)
- challenge (what is the "problem" we could solve with this dataset)
- action (how will we try to solve this problem/answer this question)
- resolution (what did we end up producing)

### Introduction

Diabetes dataset

Ten baseline variables, age, sex, body mass index, average blood pressure, and six blood serum measurements were obtained for each of n = 442 diabetes patients, as well as the response of interest, a quantitative measure of disease progression one year after baseline. 
 
Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times n_samples(i.e. the sum of squares of each column totals 1). 

https://scikit-learn.org/stable/datasets/index.html#diabetes-dataset

According to the initial revision, the information is complete, without null or missing data.   In the same way, I do not see "atypical" data.  It is understood that as a result of these steps the data is ready to start exploration and modeling.

### Methods



Brief (no more than 1-2 paragraph) description about how you decided to approach solving it. Include:

- pseudocode for this method (either created by you or cited from somewhere else)
- why you chose this method

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

Variables with p-values above 0.05 (relationship to the response variable possibly insignifcant): x1(AGE), x5(TC), x6(LDL), x7(TCH) x8( and x10(just barely), and GLU

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
