import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.io as pio
import plotly_express as px
import pandas as pd
import seaborn as sns
import time
from scipy.stats import linregress
import statsmodels.formula.api as smf
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_diabetes

data = load_diabetes()

df  = pd.DataFrame(data.data)

df.columns= data.feature_names
df['y']=data.target

#creating Dataframe from Files
print(df)
time.sleep(1)
print(df.columns)
time.sleep(1)
print(df.dtypes)
time.sleep(1)
print(df.shape)
time.sleep(1)
print(df.info())
time.sleep(1)
print(df.describe())
time.sleep(1)
print(df.corr())# -*- coding: utf-8 -*-
time.sleep(1)

os.makedirs('diabetes7plots', exist_ok=True)
plt.matshow(df.corr())
plt.xticks(range(len(df.columns)), df.columns)
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar()
plt.savefig(f'diabetes7plots/heatmapplot.png', format='png')
plt.clf()
plt.close()

# Basic correlogram
sns.pairplot(df)
plt.savefig(f'diabetes7plots/pairplot.png', format='png')
plt.clf()
plt.close()

# Plotting line chart
plt.style.use("ggplot")
plt.plot(df['bp'], color='blue', marker='o')
plt.title('BLOOD PRESSURE RANGE\nInitial info')
plt.xlabel('Samples')
plt.ylabel('Blood Pressure')
plt.savefig(f'diabetes7plots/BP_to_see_plot.png', format='png')
plt.clf()
plt.close()

# Add jittering to age
bmi = df['bmi'] + np.random.normal(0, 2.5, size=len(df))
dprogression=df['y']
# Make a scatter plot
plt.plot(bmi, dprogression, 'o', markersize=5, alpha=0.2)
plt.xlabel('BODY MASS INDEX')
plt.ylabel('DIABETES PROGRESSION AFTER 1 YEAR')
plt.savefig(f'diabetes7plots/other_regexample.png', format='png')
plt.clf()
plt.close()

# Pie
fig, axes = plt.subplots(1, 1, figsize=(8, 8))
axes.pie(df['sex'].value_counts(), labels=df['sex'].value_counts().index.tolist())
axes.set_title('GENDER')
axes.legend()
plt.savefig(f'diabetes7plots/PIE_GENDER.png', format='png')
plt.clf()
plt.close()

# Plotting TWO lines chart
fig, ax= plt.subplots()
ax.plot(df.index, df["bmi"], color='blue')
ax.set_xlabel("Samples")
ax.set_ylabel("Body Mass Index", color='blue')
ax.tick_params('y', colors='blue')
ax2 = ax.twinx()
ax2.plot(df.index, df['y'], color='green')
ax2.set_ylabel('Diabetes progression after 1 year', color='green')
ax2.tick_params('y', colors='green')
ax.set_title('Body Mass Index and Diabetes progression')
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/BMI_vs_Y_plot.png', format='png')
plt.clf()
plt.close()

# Plotting histogram
plt.hist(df['age'], bins=10, histtype='bar', rwidth=0.6, color='b')
plt.title('AGE RANGE')
plt.xlabel('AGE')
plt.ylabel('COUNT')
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/AGE_hist.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(df['age'], df['y'], color='b', marker='x', s=10)
plt.title('AGE AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('AGE')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/progression_AGE.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(df['age'], df['sex'], color='b', marker='v', s=10)
plt.title('AGE AND GENRE CORRELATION')
plt.xlabel('AGE')
plt.ylabel('GENRE')
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/CORR_AGE_GENRE.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(df['sex'], df['y'], color='g')
plt.title('GENRE AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('SEX')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/progression_SEX.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(df['bmi'], df['y'], color='g')
plt.title('BODY MASS INDEX AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('BMI')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/progression_BMI.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(df['bp'], df['y'], color='g')
plt.title('BLOOD PRESSURE AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('BP')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/progression_BP.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(df['s1'], df['y'], color='g')
plt.title('TC AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('S1')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/progression_S1.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(df['s2'], df['y'], color='g')
plt.title('LDL AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('S2')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/progression_S2.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(df['s3'], df['y'], color='g')
plt.title('HDL AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('s3')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/progression_S3.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(df['s4'], df['y'], color='g')
plt.title('TCH AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('S4')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/progression_S4.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(df['s5'], df['y'], color='g')
plt.title('LTG AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('s5')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/progression_S5.png', format='png')
plt.clf()
plt.close()

# Plotting scatterplot
plt.scatter(df['s6'], df['y'], color='g')
plt.title('GLUCOSE AND DIABETES PROGRESSION -1 YEAR-')
plt.xlabel('S6')
plt.ylabel('DIABETES PROGRESSION')
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/progression_S6.png', format='png')
plt.clf()
plt.close()

#Distribution plot by AGE
sns.set()
sns.distplot(df['age'], bins=10, kde=True)
plt.savefig('diabetes7plots/distplotage.png', format='png')
plt.clf()
plt.close()

sns.set()
for jointplot_kind in ['reg', 'hex', 'kde', 'scatter']:
    sns.jointplot('bmi', 'y', data=df, kind=jointplot_kind)
plt.savefig(f'diabetes7plots/jointplot.png', format='png')
plt.clf()
plt.close()
    
#multiple comparisons
plt.style.use("ggplot")

fig, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.grid(axis='y', alpha=0.5)
axes.scatter(df['y'], df['bmi'], color='blue')
axes.scatter(df['y'], df['bp'], color= 'green')
axes.scatter(df['y'], df['age'])
axes.set_title(f'Progression Diabetes Comparisons')
axes.legend()
plt.savefig(f'diabetes7plots/comparisons_bmi_bp_age.png', format='png', dpi=300)
plt.clf()
plt.close()

# Create a distplot of the Age
fig, ax = plt.subplots()
sns.distplot(df['age'], ax=ax,
             hist=True,
             rug=True,
             kde_kws={'shade':True})
ax.set(xlabel="Patients Age")
plt.savefig(f'diabetes7plots/Age_hist_special.png', format='png')
plt.clf()
plt.close()

# Create a plot with 1 row and 2 columns that share the y axis label
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)
# Plot the distribution of 1 bedroom apartments on ax0
sns.distplot(df['bmi'], ax=ax0)
ax0.set(xlabel="BMI")

# Plot the distribution of 2 bedroom apartments on ax1
sns.distplot(df['age'], ax=ax1)
ax1.set(xlabel="AGE")
plt.savefig(f'diabetes7plots/BMI_AGE_COMB.png', format='png')
plt.clf()

sns.boxplot(data=df,
         x='sex',
         y='y')

plt.savefig(f'diabetes7plots/boxplot_Y_SEX.png', format='png')
plt.clf()

# Create a pointplot and include the capsize in order to show bars on the confidence interval
sns.barplot(data=df,
         y='y',
         x='sex',
         capsize=.1)
plt.savefig(f'diabetes7plots/barplot_Y_SEX.png', format='png')
plt.clf()

# Create a scatter plot by disabling the regression line BMI
sns.regplot(data=df,
            y='y',
            x="bmi",
            fit_reg=False)
plt.savefig(f'diabetes7plots/scatternogres_Y_BMI.png', format='png')
plt.clf()

#Reggresion BMI Diabetes Progression
sns.set_style('whitegrid')
sns.set(color_codes=True)
sns.regplot(data=df, color='blue',
         x="bmi",
         y="y")
sns.despine()
plt.savefig(f'diabetes7plots/regression_BMI.png', format='png')
plt.clf()

# Create regression with bins BMI
sns.regplot(data=df,
            y='y',
            x="bmi",
            x_bins=10)
plt.savefig(f'diabetes7plots/regBINS_BMI.png', format='png')
plt.clf()

#POLYNOMIAL Reggresion BMI Diabetes Progression
sns.regplot(data=df, x='bmi', y='y', order=2)
sns.despine()
plt.savefig(f'diabetes7plots/polynregression_BMI.png', format='png')
plt.clf()

#POLYNOMIAL WITH BINS Reggresion BMI Diabetes Progression
sns.regplot(data=df, x='bmi', y='y', order=2, x_bins=10)
sns.despine()
plt.savefig(f'diabetes7plots/polyBINSreg_BMI.png', format='png')
plt.clf()

#Residual plot POLYNOMIAL Reggresion BMI Diabetes Progression
sns.residplot(data=df, x='bmi', y='y', order=2)
plt.savefig(f'diabetes7plots/residualpolynregression_BMI.png', format='png')
plt.clf()

#Reggresion AGE BINS Diabetes Progression
sns.set_style('whitegrid')
sns.set(color_codes=True)
sns.regplot(data=df, x='age', y='y', x_bins=10)
sns.despine()
plt.savefig(f'diabetes7plots/regressionBINS_AGE.png', format='png')
plt.clf()

#Reggresion BMI Diabetes Progression lmplot
for p in ['bright', 'colorblind']:
    sns.set_palette(p)
sns.lmplot(data=df,
         x="bmi",
         y="y")
plt.savefig(f'diabetes7plots/regressionlmplot_BMI.png', format='png')
plt.clf()

# Display the residual plot BMI regression
sns.residplot(data=df,
          y='y',
          x="bmi",
          color='g')
plt.savefig(f'diabetes7plots/residualplot_BMI.png', format='png')
plt.clf()

#heatmap seaborn
sns.heatmap(df.corr())
plt.savefig(f'diabetes7plots/heatseaborn.png', format='png')
plt.clf()

# Create a PairGrid with a scatter plot for BMI and Diabetes progression
g = sns.PairGrid(df, vars=["bmi", "y"])
g2 = g.map(plt.scatter)
plt.savefig(f'diabetes7plots/pairgrid_bmi_y_.png', format='png')
plt.clf()
plt.close()

# Create the same pairgrid but map a histogram on the diag
g = sns.PairGrid(df, vars=["bmi", "y"])
g2 = g.map_diag(plt.hist)
g3 = g2.map_offdiag(plt.scatter)
plt.savefig(f'diabetes7plots/pairgridscatter_bmi_y_.png', format='png')
plt.clf()
plt.close()

# Create the same pairgrid but map a histogram on the diag
g = sns.PairGrid(df, vars=["bmi", "y", "age"])
g2 = g.map_diag(plt.hist)
g3 = g2.map_offdiag(plt.scatter)
plt.savefig(f'diabetes7plots/pairgridscatter_bmi_y_AGE.png', format='png')
plt.clf()
plt.close()

# Plot the same data but use a different color palette and color code by Region
sns.pairplot(data=df,vars=["bmi", "y"],
        kind='scatter',
        hue='sex',
        palette='RdBu',
        diag_kws={'alpha':.5})
plt.savefig(f'diabetes7plots/pairPLOT_bmi_y_SEX.png', format='png')
plt.clf()

# Hexbin plot with bivariate distribution
sns.jointplot(x='bmi', y='y', data=df, kind='hex', height=7, color='g')
plt.savefig(f'diabetes7plots/hex_bmi_y.png', format='png')
plt.clf()
plt.close()

# KDE plot with bivariate distribution
sns.jointplot(x='bmi', y='y', data=df, kind='kde', height=7, color='g')
plt.savefig(f'diabetes7plots/KDE_BIVAR_bmi_y.png', format='png')
plt.clf()
plt.close()

# 3D for gender
female= df[df['sex'] == 1]
male= df[df['sex'] == 2]
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d')
line1 = axes.scatter(female['bmi'], female['age'], female['y'])
line2 = axes.scatter(male['bmi'], male['age'], male['y'])
axes.legend((line1, line2), ('female', 'male'))
axes.set_xlabel('BMI')
axes.set_ylabel('AGE')
axes.set_zlabel('DIABETES PROGRESSION')
plt.savefig(f'diabetes7plots/3D_BMI_AGE_Y.png', format='png')
plt.clf()
plt.close()


# Build a pairplot with different x and y variables
sns.pairplot(data=df,
        x_vars=['bmi', 'age'],
        y_vars=['y', 's6'],
        kind='scatter',
        hue='sex',
        palette='husl')
plt.savefig(f'diabetes7plots/pairPLOT_multiple.png', format='png')
plt.clf()
plt.close()

# Build a pairplot with different x and y variables
sns.pairplot(data=df,
        x_vars=['bmi'],
        y_vars=['y', 'age', 'sex', 's2', 's4', 's6'],
        kind='scatter',
        palette='husl')
plt.savefig(f'diabetes7plots/pairPLOT_multiple_6.png', format='png')
plt.clf()
plt.close()

# plot relationships between BMI and diabetes progression
sns.pairplot(data=df,
             x_vars=["bmi"],
             y_vars=["y", "age"],
             kind='reg',
             palette='BrBG',
             diag_kind = 'kde',
             hue='sex')
plt.savefig(f'diabetes7plots/pairPLOT_KDE.png', format='png')
plt.clf()
plt.close()

# Build a JointGrid comparing BMI and diabetes progression
sns.set_style("whitegrid")
g = sns.JointGrid(x="bmi",
            y="y",
            data=df)
g.plot(sns.regplot, sns.distplot)

plt.savefig(f'diabetes7plots/jointgridbasic.png', format='png')
plt.clf()
plt.close()

# Plot a jointplot showing the residuals
sns.jointplot(x="bmi",
        y="y",
        kind='resid',
        data=df,
        order=2)
plt.savefig(f'diabetes7plots/residjointpoly.png', format='png')
plt.clf()
plt.close()

# Create a jointplot of BMI vs. diabetes progression
# Include a kdeplot over the scatter plot
sns.set_style('whitegrid')
g = (sns.jointplot(x="bmi",
             y="y",
             kind='reg',
             data=df,
             marginal_kws=dict(bins=10, rug=True))
    .plot_joint(sns.kdeplot))
fig.set_size_inches([8,8])
plt.savefig(f'diabetes7plots/kdeplot_bmi_y.png', format='png')
plt.clf()
plt.close()

# Create a jointplot of GLUCOSE vs. diabetes progression
# Include a kdeplot over the scatter plot
g = (sns.jointplot(x="s6",
             y="y",
             kind='reg',
             data=df,
             marginal_kws=dict(bins=10, rug=True))
    .plot_joint(sns.kdeplot))
plt.savefig(f'diabetes7plots/kdeplot_GLU_y.png', format='png')
plt.clf()
plt.close()

# Create a jointplot of S5 vs. diabetes progression
# Include a kdeplot over the scatter plot
g = (sns.jointplot(x="s5",
             y="y",
             kind='reg',
             data=df,
             marginal_kws=dict(bins=10, rug=True))
    .plot_joint(sns.kdeplot))
plt.savefig(f'diabetes7plots/kdeplot_S5_y.png', format='png')
plt.clf()
plt.close()

# Create a jointplot of BMI VS.GLUCOSE
# Include a kdeplot over the scatter plot
g = (sns.jointplot(x="bmi",
             y="s6",
             kind='reg',
             data=df,
             marginal_kws=dict(bins=10, rug=True))
    .plot_joint(sns.kdeplot))
plt.savefig(f'diabetes7plots/kdeplot_BMI_GLU.png', format='png')
plt.clf()
plt.close()

# Create a jointplot of BMI VS.S5
# Include a kdeplot over the scatter plot
g = (sns.jointplot(x="bmi",
             y="s5",
             kind='reg',
             data=df,
             marginal_kws=dict(bins=10, rug=True))
    .plot_joint(sns.kdeplot))
plt.savefig(f'diabetes7plots/kdeplot_BMI_S5.png', format='png')
plt.clf()
plt.close()

# Create a regression plot using hue
sns.lmplot(data=df,
           x="bmi",
           y="y",
           hue="sex")
plt.savefig(f'diabetes7plots/regressionlmplot_BMI_sex.png', format='png')
plt.clf()

# Create a regression plot using hue
sns.lmplot(data=df,
           x="age",
           y="y",
           hue="sex")
plt.savefig(f'diabetes7plots/regressionlmplot_AGE_SEX.png', format='png')
plt.clf()

# Create a regression plot with multiple rows
sns.lmplot(data=df,
           x="bmi",
           y="y",
           row="sex")
plt.savefig(f'diabetes7plots/regressionMULT_SEX.png', format='png')
plt.clf()


# 2 in 1
fig, axes = plt.subplots(4, 1, figsize=(8,8))

# Plotting scatterplot
plt.style.use("ggplot")
axes[1].scatter(df['bmi'], df['bp'], color='b', marker='x', s=10)
axes[1].set_title('BMI AND AVERAGE BLOOD PRESSURE')
axes[1].set_xlabel('BMI')
axes[1].set_ylabel('BLOOD PRESSURE')
# Plotting scatterplot
plt.style.use("ggplot")
axes[0].scatter(df['bmi'], df['y'], color='b', marker='x', s=10)
axes[0].set_title('BMI AND DIABETES PROGRESSION -1 YEAR-')
axes[0].set_xlabel('BMI')
axes[0].set_ylabel('DIABETES PROGRESSION')
# Plotting scatterplot
axes[2].scatter(df['bmi'], df['s5'], color='b', marker='x', s=10)
axes[2].set_title('BMI AND S5')
axes[2].set_xlabel('BMI')
axes[2].set_ylabel('S5')
# Plotting scatterplot
axes[3].scatter(df['bmi'], df['age'], color='b', marker='x', s=10)
axes[3].set_title('BMI AND AGE')
axes[3].set_xlabel('BMI')
axes[3].set_ylabel('AGE')
plt.tight_layout()
plt.savefig(f'diabetes7plots/jointplotNEW.png', format='png')
plt.clf()
plt.close() 