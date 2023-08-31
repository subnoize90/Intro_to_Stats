# Intro_to_Stats
A collection of methods for Stats 216.

This package was designed for an Introductory Statistics course and contains prebuilt methods for basic statistical analysis. This module was made for use with course-provided materials including essential tables such as the area under the normal distribution curve, the area in the right tail of t-distribution, and critical values for the Pearson Correlation Coefficient. These values can be attained from other packages like Scipy but may yield slightly different results than course-provided tables as such it is recommended to use course-provided materials if using this package in parallel with a statistics class.

It is also worth noting that many of the functions that are built into this package are already built into popular Python packages such as Scipy and Pandas.  This project was an attempt to rebuild these functions with the working knowledge of the formulas and methods taught throughout the course. It is meant to be a resource for others who are taking an Intro to Stats course and are also trying to get a working Pythonic knowledge of statistics.

Functions included:

fact 					- get factorial 

comb 					-get combinations

perm					-get permutations 

get_µX					-get mu of x (n*p)

get_σX					-get std of x (sqrt of n*p*(1-p))

get_µX_and_σX				-get std and mu of x

get_σ_X̅				-get std of x bar (σ / sqrt of n)

get_σ_p_hat				-get std of p hat (sqrt of p*(1-p)/n)

get_z_0					-get z naught from p naught, p hat, and n

get_confidence_interval_norm-		-get confidence interval from a normal distribution

get_population_estimate_for_n		-get a population estimate based on confidence – C

get_t_value				-get t-value from scipy

get_t_confidence_interval		-get confidence interval from t-value,x bar, sample std, and n

get_sample_size_estimate		-get sample size from Standard Error and confidence Za

get_proportion_confidence_interval	-get a confidence interval for a proportion

get_t_distribution				-get T distribution (x_bar - µ / (s / sqrt of n))	

get_p_value_from_t_stat			- get p value from t stat using scipy

get_z_score					-get z score from mean std and x

big13						- use pandas series to return: count, STD, mean, 
						min/max, range, Q1,Q2,Q3,IQR,upper/lower fence
						and the variance

bin_app					- Approximate mean and STD of a series using bin range
						This method requires a ‘frequency” column to approx.

bin_freq					- use pandas series to return a frequency and relative
						frequency series in the same dataframe based on bin
						ranges

sample_combo					- returns possible sample combinations from a pandas
						series with arguments for unique sets and combinations
						with replacements 

empirical_rule					- prints off a visual representation of empirical rule 
						from mean and std

freq_counter					-creates a frequency and relative frequency series
						in the dataframe from a selected series

histogram					-return a histogram from pandas series with specified
						bins and an option for a KDE plot

suppress					-suppresses scientific notation

boxplot						- returns a boxplot of the specified pandas series
						may use multiple series in a list

barchart					-returns a barchart from series with option to set index

scatterplot					-returns scatterplot from two pandas series

Pearson_corr_coeff				- returns the Pearson Correlation Coefficient from two 
						Pandas series with option for numeric only

slope						-returns the slope from two points (P and Q)
x_and_y_intercept				-returns the x and y intercept from two points

least_squares_regression			- returns a scatterplot with least squares regression
						has options for secondary slope and y-intercept
						comparison as well as an option for testing x

farey						-converts decimal to a fraction- option for max denom

critical_values_corr_coeff			-returns a list of critical values up to n=20

probability_empirical				-returns a series of probabilities using the empirical rule

sample_space					-returns a list of the sample space from a pandas series

binomial_probability_distribution		-returns binomial probability from n, p, and x

cumulative_binomial_probability_distribution	- returns cumulative probability from n, p, and x

range_binomial_probability_dist			-returns probability from a range (x)

mean_std_discrete_randvar				-returns µ and σ from an x column and a 
							P of x column (probability of x)

kde_plot						-returns a KDE plot with a color-in option for								greater than, in between and outside the values

normal_distribution_generator				-generates a normal distribution from µ, σ,
							and size. Uses random number so- n > 10000  
							use if no data for KDE plot


This package is dependent on the following packages:
Matplotlib
Pandas
Numpy
Seaborn
Math
Scipy


This module will be updated for efficiency in the future


