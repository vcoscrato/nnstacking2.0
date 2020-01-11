import numpy as np
import pandas as pd
from scipy.stats import t

def significance(m1, m2, e1, e2, n):
    denom = np.sqrt(e1**2 + e2**2)
    num = m1 - m2
    T = num/denom
    df_num = denom**4
    df_denom = (1/(n-1))*e1**4 + (1/(n-1))*e2**4
    df = df_num/df_denom
    return t.cdf(T, df)

def sig_all(results, n, path):
    mse = [i[0] for i in results]
    mse_std = [i[1] for i in results]
    mae = [i[2] for i in results]
    mae_std = [i[3] for i in results]
    out_mse = np.empty((13, 4))
    out_mae = np.empty((13, 4))
    for i in range(4):
        for j in range(13):
            if i == j:
                out_mse[j, i] = np.nan
                out_mae[j, i] = np.nan
            else:
                out_mse[j, i] = significance(mse[i], mse[j], mse_std[i], mse_std[j], n)
                out_mae[j, i] = significance(mae[i], mae[j], mae_std[i], mae_std[j], n)
    out_mse = pd.DataFrame(out_mse)
    out_mse.columns = ['UNNS+φ(x)', 'CNNS+φ(x)', 'UNNS', 'CNNS']
    out_mse.index = ['UNNS+φ(x)', 'CNNS+φ(x)', 'UNNS', 'CNNS',
                     'Breiman’s stacking', 'Meta-regression neural net', 'Direct neural net', 
                     'Least squares', 'Lasso', 'Ridge', 'Bagging', 'Random forest', 'Gradient boosting']
    out_mae = pd.DataFrame(out_mae)
    out_mae.columns = ['UNNS+φ(x)', 'CNNS+φ(x)', 'UNNS', 'CNNS']
    out_mae.index = ['UNNS+φ(x)', 'CNNS+φ(x)', 'UNNS', 'CNNS',
                     'Breiman’s stacking', 'Meta-regression neural net', 'Direct neural net', 
                     'Least squares', 'Lasso', 'Ridge', 'Bagging', 'Random forest', 'Gradient boosting']
    with open(path+'Significance_MSE', 'w') as f:
        print(out_mse.to_latex(float_format="{:0.2f}".format), file=f)
    with open(path+'Significance_MAE', 'w') as f:
        print(out_mae.to_latex(float_format="{:0.2f}".format), file=f)
    return 'Success!'

if __name__ == '__main__':
    
    results = [[11400.43, 250.03, 45.91, 0.39],
               [19371.98, 429.96, 53.09, 0.52],
               [11335.85, 241.94, 45.85, 0.39],
               [18748.66, 424.5, 51.65, 0.52],
               [30829.11, 717.13, 62.41, 0.67],
               [24186.4, 545.52, 58.79, 0.59],
               [14595.98, 307.11, 52.3, 0.44],
               [79999.09, 1504.75, 176.41, 0.9],
               [80091.85, 1526.05, 175.5, 0.9],
               [79999.05, 1504.76, 176.41, 0.9],
               [31136.93, 737.47, 62.35, 0.67],
               [30923.64, 727.99, 62.2, 0.67],
               [32043.23, 676.1, 90.51, 0.63]]
    sig_all(results, 60400, 'Results/GPU/')
    
    results = [[92.37, 7.18, 6.53, 0.02],
               [83.05, 0.57, 6.38, 0.02],
               [95.35, 1.81, 7.45, 0.02],
               [82.99, 0.57, 6.38, 0.02],
               [87.66, 0.57, 6.61, 0.02],
               [87.64, 0.59, 6.61, 0.02],
               [1596.2, 10.88, 29.83, 0.07],
               [92.03, 0.62, 6.82, 0.02],
               [92.61, 0.62, 6.87, 0.02],
               [92.03, 0.62, 6.82, 0.02],
               [92.83, 0.59, 6.84, 0.02],
               [92.6, 0.59, 6.83, 0.02],
               [87.49, 0.6, 6.58, 0.02]]
    sig_all(results, 60400, 'Results/Year/')
    
    results = [[542.02, 62.65, 5.89, 0.2],
               [548.99, 63.9, 5.44, 0.2],
               [557.95, 61.51, 6.38, 0.2],
               [540.68, 63.87, 5.44, 0.2],
               [593.74, 73.19, 5.41, 0.21],
               [537.66, 63.31, 5.53, 0.2],
               [676.79, 81.0, 7.52, 0.22],
               [878.88, 109.42, 9.56, 0.25],
               [877.11, 108.11, 9.04, 0.25],
               [877.92, 109.47, 9.53, 0.25],
               [619.04, 88.49, 5.27, 0.21],
               [585.22, 64.88, 5.37, 0.21],
               [557.28, 63.88, 5.75, 0.2]]
    sig_all(results, 60400, 'Results/Blog/')
    
    results = [[98.97, 4.67, 5.71, 0.11],
               [98.79, 4.67, 5.65, 0.11],
               [98.62, 4.77, 5.64, 0.11],
               [98.60, 4.75, 5.60, 0.11],
               [99.79, 4.95, 5.48, 0.11],
               [99.05, 4.78, 5.60, 0.11],
               [274.93, 7.20, 7.20, 0.16],
               [308.65, 13.41, 7.12, 0.16],
               [475.6, 17.08, 9.41, 0.19],
               [309.17, 13.42, 7.17, 0.16],
               [105.14, 5.68, 5.02, 0.12],
               [103.02, 5.59, 5.08, 0.12],
               [161.48, 8.74, 5.05, 0.13]]
    sig_all(results, 60400, 'Results/Superconductivity/')