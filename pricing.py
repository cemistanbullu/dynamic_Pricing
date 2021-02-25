import pandas as pd
from scipy.stats import shapiro
import scipy.stats as stats
import itertools
import statsmodels.stats.api as sms


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


df_ = pd.read_csv("pricing.csv", sep=";")
df = df_.copy()

df.head()
df.info

df.describe([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T
df.shape
df["category_id"].nunique()
df.isnull().sum()

df.groupby("category_id").agg({"price": ["median"]})


# To determine the threshold value for outliers:

def outlier_thresholds(dataframe, variable, low_quantile=0.05, up_quantile=0.95):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


def has_outliers(dataframe, numeric_columns):
    for col in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, " : ", number_of_outliers, "outliers")


def remove_outliers(dataframe, numeric_columns):
    for variable in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe_without_outliers = dataframe[~((dataframe[variable] < low_limit) | (dataframe[variable] > up_limit))]
    return dataframe_without_outliers


remove_outliers(df, ["price"])
has_outliers(df, ["price"])

# H0 : There is no statistically significant difference between price average of two category
# H1 : There is statistically significant difference between price average of two category
# Assumptions of the Hypothesis:

############################
# Normal Distribution
############################

# H0: There is no statistically significant difference between sample distribution and theoretical normal distribution
# H1: There is statistically significant difference between sample distribution and theoretical normal distribution

print(" Shapiro-Wilks Test Result")
for category in df["category_id"].unique():
    test_statistic, pvalue = shapiro(df.loc[df["category_id"] == category, "price"])
    if (pvalue < 0.05):
        print('\n', '{0} -> '.format(category), 'Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue),
              "H0 is rejected.")
    else:
        print('Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue), "H0 is not rejected.")

# Normal distribution was not achieved




# Non-parametric (mannwhitneyu)

# Test Statistic:
# H0: There is no difference between the median price paid for the two categories.
# H1: There is a difference between the median price paid for the two categories.

pairs = []

for i in itertools.combinations(df["category_id"].unique(), 2):
    pairs.append(i)
pairs
for i in pairs:
    test_statistic, pvalue = stats.mannwhitneyu(df.loc[df["category_id"] == i[0], "price"],
                                                df.loc[df["category_id"] == i[1], "price"])
    if pvalue < 0.05:

        print(f"{i[0]} - {i[1]} -> ", "test_statistic = %.4f, p-value = %.4f" % (test_statistic, pvalue),
              "H0 is rejected")
    else:

        print(f"{i[0]} - {i[1]} -> ", "test_statistic = %.4f, p-value = %.4f" % (test_statistic, pvalue),
              "H0 is not rejected")



print('test_statistic = %.4f, p-value = %.4f' % (test_statistic, pvalue))

# 489756 is different
# 326584 is different
# Categorical groups with no statistically significant difference : 874521, 675201, 201436, 361254

# Does the price of the item differ by category?
# When we examine the table above, there is no statistically significant difference average price between
# 5 categorical pairs, while there is a statistically significant difference average price between 10 categorical pairs.

# 2- What should the item cost?

# Median values can be determined as prices separately for 489756 and 326584.
df.loc[df["category_id"] == 489756, "price"].median()
# price: 35.635784234

df.loc[df["category_id"] == 326584, "price"].median()
# price: 31.7482419128


# Average median value for others can be determined as price
fancy = [675201, 201436, 874521, 361254]

top = []
for i in fancy:
    top.append(df.loc[df["category_id"] == i, "price"].median())

# Median price of those not different:
sum(top) / len(top)
# 34.057574562275

# 3- Confidence Intervals: It is desirable to be "flexible" in terms of price


sms.DescrStatsW(top).tconfint_mean()

# 4- Simulation For Item Purchases

# Median values can be set as prices separately for 489756 and 326584
# 489756
freq = len(df[df["price"] >= 35.635784234])  # number of sales equal to or greater than this price
income = freq * 35.635784234  # income
print("Income: ", income)

# 326584
freq = len(df[df["price"] >= 31.7482419128])  # number of sales equal to or greater than this price
income = freq * 31.7482419128  # income
print("Income: ", income)

# conf interval min
freq = len(df[df["price"] >= 33.344860323448636])  # number of sales equal to or greater than this price
income = freq * 33.344860323448636  # income
print("Income: ", income)
# conf interval max
freq = len(df[df["price"] >= 34.7702888011013])  # number of sales equal to or greater than this price
income = freq * 34.7702888011013  # income
print("Income: ", income)

# average : 34.057574562275
freq = len(df[df["price"] >= 34.057574562275])  # number of sales equal to or greater than this price
income = freq * 34.057574562275  # income
print("Income: ", income)
