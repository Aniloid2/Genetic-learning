import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, FuncFormatter, StrMethodFormatter, FixedFormatter
import math


n = 21


# If told so it can plot the landscapes used for fig1 in the simple 2 module problem report.
# In the main program, the generate matrix function is called to create a landscape 
class Plots:

    def __init__(self, bits):
        self.bits = bits
        self.bits_expanded = [i for i in range(bits)]
        self.Noise = self.return_noise()
    
    def initaliaze(self):
        I, i, j = self.generate_matrix('Ideal')
        R, i, j = self.generate_matrix('Rugged')
        
        self.plot_noise(self.Noise)
        self.plot(I, 'ideal.png')
        self.plot(R, 'rugged.png')

        self.show()

    def return_noise(self):
        R = np.random.uniform(0.5,1,size=(self.bits,self.bits))
        return R


    def generate_matrix(self, function):
        M = np.zeros((self.bits,self.bits))
        maximum_found_fitness = 0
        global_i = 0
        global_j = 0 
        for i in range(self.bits):
            Row = np.zeros(self.bits)
            for j in range(self.bits):
                if function == 'Ideal':
                    L = 2**(i)+2**(j)
                elif function == 'Rugged':
                    L = self.Noise[i,j]*(2**(i)+2**(j))
                else:
                    print ('Parse eather Ideal, or rugged as a parameter')
                if maximum_found_fitness < L:
                    maximum_found_fitness = L
                    global_i = i
                    global_j = j
                Row[j] = L
            M[i:] = Row
        return (M, global_i, global_j)

    def plot(self, Z, which ,format_axis = True):  
        fig = plt.figure()
        X, Y = np.meshgrid(self.bits_expanded, self.bits_expanded)
        ax = fig.add_subplot(111, projection='3d')
        if format_axis == True:
            ax.set_zlim(0, 2500000)
            ax.set_xlim(0, 20)
            ax.set_ylim(0, 20)
            ax.zaxis.set_major_formatter(FuncFormatter(lambda x, pos: '%dk' % int( x/1000 )))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        
        ax.view_init(35, 230)
        ax.set_xlabel('i')
        ax.set_ylabel('j')
        ax.set_zlabel('Fitness')
        ax.zaxis.labelpad=15
        ax.zaxis.set_rotate_label(False)
        Axes3D.plot_surface(ax,X,Y,Z, cmap=cm.coolwarm)
        fig.savefig(which)

    def plot_noise(self,Z):
        fig = plt.figure()
        X, Y = np.meshgrid(self.bits_expanded, self.bits_expanded)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(0, 1.1)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.view_init(35, 230)
        ax.set_xlabel('i')
        ax.set_ylabel('j')
        ax.set_zlabel('R(i,j)')
        ax.zaxis.set_rotate_label(False)
        Axes3D.plot_surface(ax,X,Y,Z, cmap=cm.coolwarm)
        fig.savefig('noise.png')


    def show(self):
        plt.show()




Plots(n).initaliaze()