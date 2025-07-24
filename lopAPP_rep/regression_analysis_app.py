
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy.linalg as la

from sklearn.linear_model import LinearRegression
from linearmodels import PooledOLS
from linearmodels import PanelOLS
from linearmodels.panel import RandomEffects
from linearmodels.panel import compare
from scipy import stats
from typing import Tuple


def hausman(fe, re):
    b = fe.params
    B = re.params
    v_b = fe.cov
    v_B = re.cov

    df = b[np.abs(b) < 1e8].size
    chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B))

    pval = stats.chi2.sf(chi2, df)
    return chi2, df, pval


# hausman_results = hausman(fe, panel_reg_re)
# print('chi-Squared: ' + str(hausman_results[0]))
# print('degrees of freedom: ' + str(hausman_results[1]))
# print('‘p-Value: ' + str(hausman_results[2]))

# Function is more general, improves readability of code


def load_data(file_name, columns, index_columns=['Country', 'Year']):
    df = pd.read_csv(file_name, usecols=columns, index_col=index_columns)
    return df


# Load the data
dftc_columns = ['Year', 'Country', 'VATGST',
                'TARIFFipod', 'TARIFFipad', 'TARIFFiphone', 'SHIP']
dfcpi_columns = ['Year', 'Country', 'CPIave', 'CPIER']
dfbmi_columns = ['Year', 'Country', 'BigMacD', 'BigMacER', 'VATGST']
dfipod_columns = ['Year', 'Country', 'iPodD',
                  'iPodER', 'VATGST', 'TARIFFipod', 'SHIP']
dfipad_columns = ['Year', 'Country', 'iPadD',
                  'iPadER', 'VATGST', 'TARIFFipad', 'SHIP']
dfiphone_columns = ['Year', 'Country', 'iPhoneD',
                    'iPhoneER', 'VATGST', 'TARIFFiphone', 'SHIP']


dftc = load_data('appsannual.csv', dftc_columns)
dfcpi = load_data('appsannual.csv', dfcpi_columns)
dfbmi = load_data('appsannual.csv', dfbmi_columns)
dfipod = load_data('appsannual.csv', dfipod_columns)
dfipad = load_data('appsannual.csv', dfipad_columns)
dfiphone = load_data('appsannual.csv', dfiphone_columns)

years = list(range(2007, 2021))

for year in years:
    cpi_base_col = f'CPIBase{year}'
    cpi_ratio_col = f'CPIRatio{year}'
    lni_cpi_ratio_col = f'lniCPIRatio{year}'

    dfcpi[cpi_base_col] = dfcpi['CPIave'] / \
        dfcpi.xs(year, level=1)['CPIave'] * dfcpi.xs(year, level=1)['CPIER']
    dfcpi[cpi_ratio_col] = dfcpi[cpi_base_col] / \
        dfcpi.loc['United States'][cpi_base_col]
    dfcpi[lni_cpi_ratio_col] = np.log(dfcpi[cpi_ratio_col])

dfcpi = (
    dfcpi.assign(lnCPIER=lambda x: np.log(x['CPIER']))
    .sort_index(axis=0)
    .drop('Argentina', axis=0)
)


# function to run the specific model that you want to use
def run_regression(model_type: str, dep, exog) -> Tuple[float, float, int, float, float, float]:
    if model_type == 'PooledOLS':
        mod = PooledOLS(dep, exog)
    elif model_type == 'PanelOLS_efe':
        mod = PanelOLS(dep, exog, entity_effects=True)
    elif model_type == 'PanelOLS_tfe':
        mod = PanelOLS(dep, exog, time_effects=True)
    elif model_type == 'PanelOLS_etfe':
        mod = PanelOLS(dep, exog, entity_effects=True, time_effects=True)
    elif model_type == 'RandomEffects':
        mod = RandomEffects(dep, exog)
    else:
        raise ValueError("Invalid model_type")

    result = mod.fit()

    return result.params[0], result.params[1], result.nobs, result._r2o, result.std_errors[0], result.std_errors[1]


# def print_results(model_type: str, year: int, result: Tuple[float, float, int, float, float, float]):
#     print(
#         f"\n\multicolumn{{1}}{{l}}{{{model_type} \\footnotesize{{CPI Base {year}}}}}\n")
#     print(
#         f"&  {result[0]:.4f} & {result[1]:.4f} && {result[2]} & {result[3]:.4f}{chr(92)}{chr(92)}")
#     print(
#         f"& \\footnotesize{{({result[4]:.4f})}} & \\footnotesize{{({result[5]:.4f})}} &&& {chr(92)}{chr(92)}\n")


years = list(range(2007, 2021))
model_types = ['PooledOLS']
# model_types = ['PooledOLS', 'PanelOLS_efe', 'PanelOLS_tfe', 'PanelOLS_etfe', 'RandomEffects']

for year in years:
    print(year)
    exog = sm.tools.tools.add_constant(dfcpi['lnCPIER'])
    dep = dfcpi[f'lniCPIRatio{year}']

    for model_type in model_types:
        result = run_regression(model_type, dep, exog)
        # print_results(model_type, year, result)


# def print_hausman_results(year, hausman_results):
#     print(f"\nYear: {year}")
#     print("chi-Squared:", hausman_results[0])
#     print("degrees of freedom:", hausman_results[1])
#     print("p-Value:", hausman_results[2])
#     print("")

# Hausman test for CPI regressions


for year in years:
    exog = sm.tools.tools.add_constant(dfcpi['lnCPIER'])
    dep = dfcpi['lniCPIRatio' + str(year)]

    # Fixed effects model
    mod_fe = PanelOLS(dep, exog, entity_effects=True, time_effects=True)
    panel_reg_fe = mod_fe.fit()

    # Random effects model
    mod_re = RandomEffects(dep, exog)
    panel_reg_re = mod_re.fit()

    # Perform Hausman test and print results
    hausman_results = hausman(panel_reg_fe, panel_reg_re)
    # print_hausman_results(year, hausman_results)


# This is where you need to provide a breakdown of the different regressions that you are doing.


# Big Mac Price Basket Regression Results

# h-period analysis should maybe be in a separate document
