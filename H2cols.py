import numpy as np
from scipy.interpolate import Rbf
from scipy.interpolate import RectBivariateSpline

from scipy.interpolate import interp2d
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from spectro.stats import distr2d


#names = ['temp/H2j0','temp/H2j1','temp/H2j2','temp/H2j3','temp/H2j4']
#for el in names:
if 0:
    for k,e in enumerate(['H2j0','H2j1','H2j2','H2j3','H2j4','H2j5']):
        str = ('temp/',e)
        str = "".join(str)
        str_nodes = ('temp/', e,'_nodes')
        str_nodes = "".join(str_nodes)
        with open(str, 'rb') as f:
            XI, YI, ZI = pickle.load(f)
        with open(str_nodes, 'rb') as f:
            x, y, z = pickle.load(f)
        # n = plt.normalize(10., 20.)

        if 1:
            XI, YI = np.meshgrid(XI, YI)
            plt.subplot(3, 2, k+1)
            plt.pcolor(XI, YI, ZI, cmap=cm.jet, vmin=12, vmax=20)
            plt.scatter(x, y, 50, z, cmap=cm.jet, vmin=12, vmax=20)  # ,edgecolors='black')
            plt.title(''.join(('RBF interpolation of',e)))
            plt.xlim(-1.2, 3.2)
            plt.ylim(0.8, 5.2)
            plt.colorbar()
            # plt.savefig('rbf2d.png')
    plt.show()
    #plt.savefig('rbf2d.png')

if 0:
    with open('temp/H2j5', 'rb') as f:
        XI,YI,ZI = pickle.load(f)
    with open('temp/H2j5_nodes', 'rb') as f:
        x,y,z = pickle.load(f)
    #n = plt.normalize(10., 20.)

    if 1:
        XI, YI = np.meshgrid(XI,YI)
        plt.subplot(1, 1, 1)
        plt.pcolor(XI,YI, ZI, cmap=cm.jet, vmin=12, vmax=20)
        plt.scatter(x,y, 100, z, cmap=cm.jet, vmin=12, vmax=20) #,edgecolors='black')
        plt.title('RBF interpolation - testing')
        plt.xlim(-1.2, 3.2)
        plt.ylim(0.8, 5.2)
        plt.colorbar()
        #plt.savefig('rbf2d.png')
        plt.show()
    if 0:
        xi = np.linspace(-1,3,90)
        yi = np.linspace(1,5,90)
        X1, Y1 = np.meshgrid(xi,yi)

        if 1:
            # use rBF
            #rbf = Rbf(x,y,z,function='gaussian',epsilon=2)
            rbf = Rbf(x,y,z,function='multiquadric',smooth=0.1)
            #rbf = Rbf(x, y, z, function='inverse', epsilon=0.14)
            Z1 = rbf(X1,Y1)
            print(rbf(1.0,1.0))
        if 0:
            interp_spline = interp2d(x, y, z, kind='cubic')
            Z1 = interp_spline(xi,yi)

        #plot
        plt.subplot(1, 1, 1)
        plt.pcolor(X1, Y1, Z1, cmap=cm.jet, vmin=13, vmax=20)
        plt.scatter(x, y, 100, z, cmap=cm.jet, vmin=13, vmax=20) #,edgecolors='black')
        plt.title('RBF interpolation - test')
        plt.xlim(-1, 3)
        plt.ylim(1, 5)
        plt.colorbar()
        print('stop')
        #plt.savefig('rbf2d.png')
        plt.show()

if 1:
    name = 'B0405-4418_0'
    str = ('result/all/', name,'.pkl')
    str = "".join(str)
    strnodes = ('result/all/nodes/', name,'.pkl')
    strnodes = "".join(strnodes)
    with open(strnodes, 'rb') as f:
        x1,y1,z1 = pickle.load(f)
    with open(str, 'rb') as f:
        x, y, z = pickle.load(f)
        X, Y = np.meshgrid(x, y)
        plt.subplot(1, 2, 1)
        plt.pcolor(X, Y, z, cmap=cm.jet, vmin=-50, vmax=0)
        plt.scatter(x1, y1, 100, z1, cmap=cm.jet, vmin=-50, vmax=0)  # ,edgecolors='black')
        plt.title('RBF interpolation - likelihood')
        plt.xlim(-1.2, 3.2)
        plt.ylim(0.8, 5.2)
        plt.colorbar()

        d = distr2d(x=x, y=y, z=np.exp(z))
        dx, dy = d.marginalize('y'), d.marginalize('x')
        dx.stats(latex=2, name='log UV')
        dy.stats(latex=2, name='log n')
        #d.plot(color=None)
        ax = plt.subplot(1, 2, 2)
        d.plot_contour(ax=ax, color='lime', xlabel='$\log n$ [cm$^{-3}$]', ylabel='$\log I_{UV}$ [Draine field]')

        plt.show()

if 0:
    with open('temp/lnL_nodes.pkl', 'rb') as f:
        x1, y1, z1 = pickle.load(f)
    with open('temp/lnL.pkl', 'rb') as f:
        x, y, z = pickle.load(f)
        X, Y = np.meshgrid(x, y)
        plt.subplot(1, 3, 1)
        plt.pcolor(X, Y, z, cmap=cm.jet, vmin=-50, vmax=0)
        plt.scatter(x1, y1, 100, z1, cmap=cm.jet, vmin=-50, vmax=0)  # ,edgecolors='black')
        plt.title('RBF interpolation - likelihood')
        plt.xlim(-1.2, 3.2)
        plt.ylim(0.8, 5.2)
        plt.colorbar()
        # plt.savefig('rbf2d.png')
    with open('temp/lnL_nodes_0.2.pkl', 'rb') as f:
        x1, y1, z1 = pickle.load(f)
    with open('temp/lnL_0.2.pkl', 'rb') as f:
        x, y, z = pickle.load(f)
        X, Y = np.meshgrid(x, y)
        plt.subplot(1,3, 2)
        plt.pcolor(X, Y, z, cmap=cm.jet, vmin=-50, vmax=0)
        plt.scatter(x1, y1, 100, z1, cmap=cm.jet, vmin=-50, vmax=0)  # ,edgecolors='black')
        plt.title('RBF interpolation - likelihood')
        plt.xlim(-1.2, 3.2)
        plt.ylim(0.8, 5.2)
        plt.colorbar()
        # plt.savefig('rbf2d.png')
    with open('temp/lnL_nodes_0.3.pkl', 'rb') as f:
        x1, y1, z1 = pickle.load(f)
    with open('temp/lnL_0.3.pkl', 'rb') as f:
        x, y, z = pickle.load(f)
        X, Y = np.meshgrid(x, y)
        plt.subplot(1, 3, 3)
        plt.pcolor(X, Y, z, cmap=cm.jet, vmin=-50, vmax=0)
        plt.scatter(x1, y1, 100, z1, cmap=cm.jet, vmin=-50, vmax=0)  # ,edgecolors='black')
        plt.title('RBF interpolation - likelihood')
        plt.xlim(-1.2, 3.2)
        plt.ylim(0.8, 5.2)
        plt.colorbar()
        # plt.savefig('rbf2d.png')
        plt.show()

print('exit')