#%% Dependencias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.svm import SVR as svr

def partpred(pred, y, x, partitions=10):
    out = np.zeros((partitions, partitions))
    global grid1, grid2
    grid1 = np.linspace(np.min(x[:,0]), np.max(x[:,0]), num = partitions+1)
    grid2 = np.linspace(np.min(x[:,1]), np.max(x[:,1]), num = partitions+1)
    
    for i in range(0, partitions):
        select1 = np.logical_and(x1 > grid1[i], x1 <= grid1[i+1])
        for j in range(0, partitions):
            select2 = np.logical_and(x2 > grid2[j], x2 <= grid2[j+1])
            err = y[np.logical_and(select1, select2)] - pred[np.logical_and(select1, select2)]
            if len(err) > 0:
                out[i,j] = np.mean(err**2)
    
    return(out)

#%% Dados Teste    
x1 = np.random.gamma(2, 1, size=10000)
x2 = np.random.gamma(1, 1, size=10000)
y = x1 + 2*x2

f = plt.figure()
plt.plot(x1, y, 'r.', label = 'x1')
plt.plot(x2, y, 'g.', label = 'x2')
plt.xlabel('x values')
plt.ylabel('y values')
plt.legend(loc = 'lower right')
plt.show()
f.savefig('testdata.pdf', bbox_inches='tight')

#%% Modelos
x = np.column_stack((x1, x2))
randomforest = rfr().fit(x, y)
randomforest.predicted = randomforest.predict(x)

svm = svr().fit(x, y)
svm.predicted = svm.predict(x)

#%% Performance
f = plt.figure()
plt.plot(randomforest.predicted, y, 'b.', label = 'g1(x)')
plt.plot(svm.predicted, y, 'y.', label = 'g2(x)')
plt.plot([0,25], [0,25], 'r-', label = 'identity')
plt.xlabel('predicted values')
plt.ylabel('y values')
plt.legend(loc = 'lower right')
plt.show()
f.savefig('performance.pdf', bbox_inches='tight')

randomforest.performance = partpred(randomforest.predicted, y, x, 10)
svm.performance = partpred(svm.predicted, y, x, 10)
print(pd.DataFrame(randomforest.performance))

#%% Plot que vai fazer todo mundo entender tudo
fig, ax = plt.subplots()
ax.set_yticks(grid2)
ax.set_yticklabels(np.round(grid2, 1))
ax.set_xticks(grid1)
ax.set_xticklabels(np.round(grid1, 1))
ax.set_xlabel('x1')
ax.set_ylabel('x2')
for i in range(0, len(grid1)-1):
    for j in range(0, len(grid2)-1):
        if randomforest.performance[i,j] < svm.performance[i,j]:
            ax.add_patch(patches.Rectangle((grid1[i], grid2[j]), grid1[1] - grid1[0], grid2[1] - grid2[0], facecolor = 'blue', edgecolor = 'black'))
        else: 
            if randomforest.performance[i,j] > svm.performance[i,j]:
                ax.add_patch(patches.Rectangle((grid1[i], grid2[j]), grid1[1] - grid1[0], grid2[1] - grid2[0], facecolor = 'yellow', edgecolor = 'black'))
            else:
                ax.add_patch(patches.Rectangle((grid1[i], grid2[j]), grid1[1] - grid1[0], grid2[1] - grid2[0], facecolor = 'white', edgecolor = 'black'))
ax.legend([patches.Patch(facecolor='blue', edgecolor='black'), patches.Patch(facecolor='yellow', edgecolor='black'),
patches.Patch(facecolor='white', edgecolor='black')], ['g1(x) is better', 'g2(x) is better', 'no data'])


    
fig.savefig('clap.pdf', bbox_inches='tight')
        

