## -
import numpy as np
from matplotlib.pyplot import subplot, plot, show, clf, vlines
import matplotlib.pyplot as plt

# conjunto de dados {(x,y)}

means0, stds0 = [-0.4, 0.1], [1.9,  0.2]
means1, stds1 = [1.9, -0.7], [0.3, 0.4]
m = 200


def make_points(means, stds, m):
    xs = np.zeros((1,0))
    list_x = [
        np.random.randn(m, 1) * std + mean
        for mean, std in zip(means, stds)
    ]
        
    xs = np.concatenate(list_x, axis=1)
    print(xs.shape)        
    return xs

x1s = make_points(means1, stds1, m//2)
x0s = make_points(means0, stds0, m//2)

xs = np.vstack((x1s, x0s))
ys = np.hstack(( np.ones(m//2), np.zeros(m//2)))

plot(x1s[:,0], x1s[:,1], '.')
plot(x0s[:,0], x0s[:,1], '.')
show()


