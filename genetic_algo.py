import numpy as np

import matplotlib.pyplot as plt

plt.switch_backend('agg')

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, FuncFormatter, StrMethodFormatter, FixedFormatter
import math
import Plot_fitness_function as PFF
import sys
from keras.layers import Input, Dense
from keras.models import Model
from keras.backend import clear_session
import time


# Create the test, each island is held in a 'list' called graph. Each island is a sub_popoation object. Where in each sub_popolation object a list holds all the individuals as objects
# the individuals have I = list of nucleotides, fittness
class Popolation:

    PLX = False
    N_inidividuals = 0
    genome_lenght = 0
    found = False
    PLOTS = 0
    R = 0
    global_i = 0
    global_j = 0
    random_genetic_map_Bstring = []

    def __init__(self, N_islands, N_inidividuals, genome_lenght):
        self.set_genome_lenght(genome_lenght)
        self.set_N_individuals(N_inidividuals)
        self.set_random_genetic_map()
        self.set_R()
        self.pop = self.create_graph(N_islands)
        self.models = []
        

        if Popolation.PLX == True:
            Popolation.PLOTS.plot(Popolation.R, format_axis = False)
            Popolation.PLOTS.show()

    def __del__(self):
        pass

    def create_graph(self, N_islands ):
        self.graph = []
        for i in range(N_islands):
            self.graph.append(Sub_Popolation(i))


    def calculate_every_individuals_fitness(self):
        for island in self.graph:
            for island_sub_pop in range(len(island.individuals)):
                indiv = island.individuals[island_sub_pop]
                fit = indiv.calc_fittness(indiv.I)
                indiv.fitness = fit


                if indiv.fitness == Popolation.R[(Popolation.global_i),(Popolation.global_j)]:
                    Popolation.found = True




    def migration(self, n_migrations):
        
        # go throw every island, pick and individual at random, remove it and place it in the migration island
        for i in range(n_migrations):
            migration_island = []
            for island in self.graph:
                np.random.shuffle(island.individuals)
                Random_individual_from_island = island.individuals.pop()
                migration_island.append(Random_individual_from_island)

            np.random.shuffle(migration_island)

            for island in self.graph:
                island.individuals.append(migration_island.pop())


    def mutation_only(self):
        for island in self.graph:
            coral_reef_volcano = [] 
            for i in range(len(island.individuals)):
                pearent = self.wheel_selection(island.individuals)
                pearent.mutate()
                coral_reef_volcano.append(pearent)
            island.individuals = coral_reef_volcano 

    def cross_over_only(self):
        for island in self.graph:
            coral_reef_volcano = [] 
            for i in range(len(island.individuals)):
                pearent1 = self.wheel_selection(island.individuals)
                pearent2 = self.wheel_selection(island.individuals)
                cut_at = np.random.randint(1, len(pearent1.I))
                first_half = pearent1.I[:cut_at]
                second_half = pearent2.I[cut_at:]
                child = Individual()
                child.I = np.concatenate((first_half , second_half), axis=0)
                coral_reef_volcano.append(child)
            island.individuals = coral_reef_volcano

    def uniform_cross_over(self):
        for island in self.graph:
            coral_reef_volcano = [] 
            for i in range(len(island.individuals)):
                pearent1 = self.wheel_selection(island.individuals)
                pearent2 = self.wheel_selection(island.individuals)
                child = []
                for k in range(len(pearent1.I)):
                    From_who = np.random.randint(0,1)
                    if From_who == 0:
                        child.append(pearent1.I[k])
                    else:
                        child.append(pearent2.I[k])
                C = Individual()
                C.I = np.array(child) 
                C.mutate()
                coral_reef_volcano.append(C)

            island.individuals = coral_reef_volcano

    def dense_generation(self, n):

        for island in range(len(self.graph)):

            self.new_popolation = []
            self.models[island].fit_generator(self.dense_generator(self.graph[island],island, n), epochs=1, steps_per_epoch=len(self.graph[island].individuals), workers = 0, verbose=0)
            self.graph[island].individuals = self.new_popolation






    def dense_generator(self, island,island_index, n):

        while True:

            pearent1 = self.wheel_selection(island.individuals)
            child = Individual()
            P1 = np.reshape(pearent1.I,(1,n))


            child_I = self.models[island_index].predict(P1)
            child_I = np.reshape(child_I, (n,))
            C_temp = np.zeros(len(child_I))
            for i in range(len(child_I)):
                if child_I[i] >= 0.5:
                    C_temp[i] = 1
                elif child_I[i] < 0.5:
                    C_temp[i] = 0
                else:
                    print('Error the output has to be > 0 and < 1')
                    sys.exit()
            child_I = C_temp
       

            child.I = np.array(child_I)
            print (child.I)
            self.new_popolation.append(child)

            what_we_want = np.zeros(n)
            the_wanted_i = np.zeros(int(n/2))
            the_wanted_j = np.zeros(int(n/2))

            for i in range(Popolation.global_i):
                the_wanted_i[i] = 1
            for j in range(Popolation.global_j):
                the_wanted_j[j] = 1

            np.random.shuffle(the_wanted_i)
            np.random.shuffle(the_wanted_j)


            what_we_want[:int(n/2)] = the_wanted_i
            what_we_want[int(n/2):] = the_wanted_j 

            www_child = np.reshape(what_we_want, (1,n))
            yield P1, www_child




    def build_models(self, n):

        for island in self.graph:

            input_img = Input(shape=(n,))
            # "encoded" is the encoded representation of the input
            print ('model input shape', input_img.shape)
            encoded = Dense(int(n/2), activation='relu')(input_img)
            # "decoded" is the lossy reconstruction of the input
            decoded = Dense(n, activation='sigmoid')(encoded)

            # this model maps an input to its reconstruction
            autoencoder = Model(input_img, decoded)
            autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
            self.models.append(autoencoder)



    def wheel_selection(self, individuals):
        individuals.sort(key=lambda x: x.fitness, reverse=True)
        S= sum(I.fitness for I in individuals )
        Win = 0
        R = np.random.randint(0,S, dtype=np.int64)
        sum_p = 0
        for i in range(len(individuals)):
            sum_p += individuals[i].fitness
            if sum_p > R:
                Win = individuals[i]
                break

        return Win



    def set_N_individuals(self, N):
        Popolation.N_inidividuals = N

    def set_genome_lenght(self,N):
        Popolation.genome_lenght = N

    def set_R(self):
        Popolation.PLOTS = PFF.Plots(int(Popolation.genome_lenght/2 + 1))
        Popolation.R, Popolation.global_i, Popolation.global_j = Popolation.PLOTS.generate_matrix('Rugged')

    def set_random_genetic_map(self):
        Popolation.random_genetic_map_Bstring = [i for i in range(Popolation.genome_lenght)]
        np.random.shuffle(Popolation.random_genetic_map_Bstring) 

class Sub_Popolation:
    def __init__(self, Sub_pop_Id):
        self.Id = Sub_pop_Id
        self.individuals = []
        self.initialise_pop() 

    def initialise_pop(self):

        for i in range(Popolation.N_inidividuals):
            I = Individual()
            I.initialise_create_I()
            self.individuals.append(I)


class Individual:

    Random_genetic_map = False


    def __init__(self):
        self.fitness = 0
        self.I = []
        

    def initialise_create_I(self):
        self.I = np.zeros(Popolation.genome_lenght) 
        for i in range(Popolation.genome_lenght):
            self.I[i] = np.random.randint(0,2)

    def initialise_create_I_zeros(self):
        self.I = np.zeros(Popolation.genome_lenght)
        print (self.I)
        return (self.I)


    def mutate(self):
        temp_I = np.zeros(len(self.I))
        for i in range(len(self.I)):
            if np.random.uniform(0, 1) < (1/len(self.I)):
                temp_I[i] = int(np.random.randint(0,2))
            else:
                temp_I[i] = int(self.I[i])
        temp_fittness = self.calc_fittness(temp_I)
        if temp_fittness > self.fitness:
            self.I = temp_I



    def calc_fittness(self, genome):
        if Individual.Random_genetic_map == True:
            temp_genome = np.zeros(len(genome))
            for i in range(len(genome)):
                index = Popolation.random_genetic_map_Bstring[i]
                temp_genome[index] = genome[i]
            genome = temp_genome


        try:
            I_genome = genome[:int(Popolation.genome_lenght/2)]
            J_genome = genome[int(Popolation.genome_lenght/2):]
        except Exception as e:
            print (e, 'YOU CANT HAVE AN ODD GENOME')
            sys.exit()
        sum_I = int(sum(I_genome))
        sum_J = int(sum(J_genome))
        fitness = Popolation.R[sum_I,sum_J]
        return fitness



class Simulation():
    play = ['neural network', 'one point crossover']
    n = 0
    x_axis = [] 
    avaiable_colours = ['r', 'b', 'g', 'm']

    def __init__(self):
        pass
        


    def plot_graph(self):

        plot = PLOT()
        plot.generate_graph()
        plot.draw_stencil()
        print ('generating neural network only ')
        plot.graph_line(is_log = False)
        Simulation.fig.savefig('neural_network.png')
        print ('First done')


        plot = PLOT()
        plot.generate_graph()
        plot.draw_stencil_time()
        print ('generating neural network times ')
        plot.graph_line_time(is_time=True)
        Simulation.fig.savefig('neural_network_times.png')
        print ('Second done')

        plot = PLOT()
        plot.generate_graph()
        plot.draw_stencil()
        print ('generating neural network logonly ')
        plot.graph_line(is_log = True)
        Simulation.fig.savefig('neural_network_log.png')
        print ('third done')

        plot = PLOT()
        plot.generate_graph()
        plot.draw_stencil_time()
        print ('generating neural network times log ')
        plot.graph_line_time(is_log=True, is_time = True)
        Simulation.fig.savefig('neural_network_times_log.png')
        print ('forth done')

    def show(self):
        plt.show()


# To show the results create a plot, for each plot it is possible to create a line. The generate graph function is the function that generates the popolation and dose the wanted crossover
# when a Line() object is created and initialised with the right parameters.

class PLOT():
    def __init__(self):
        self.lines_to_plot = []

    def generate_graph(self):
        for number_of_lines in range(len(Simulation.play)):
            use = Simulation.play[number_of_lines] 
            L = Line()
            if use == 'one point crossover':
                L.n = 80
                
            elif use == 'mutation only':
                L.n = 50
            elif use == 'uniform crossover':
                L.n = 40
            elif use == 'randomised genetic map':
                L.n = 50
            elif use == 'neural network':
                L.n = 80

            L.x_axis = [x for x in range(10,L.n,10)]

            L.generate_line_values(use)
            self.lines_to_plot.append(L)

    def draw_stencil(self):
        Simulation.fig = plt.figure()
        self.ax = Simulation.fig.add_subplot(111)
        self.ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        self.ax.set_xlim(0, 90)
        self.ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        self.ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        
        self.ax.set_xlabel('n')
        self.ax.set_ylabel('generations to peak')


    def graph_line(self,is_log = False):
        for i in range(len(self.lines_to_plot)):
            if is_log == True:
                self.ax.semilogy(self.lines_to_plot[i].x_axis, self.lines_to_plot[i].means, '-o', label=Simulation.play[i], c=Simulation.avaiable_colours[i], alpha=0.3) # main_point
                self.ax.set_ylim(0,10000)
            else: 
                self.ax.plot(self.lines_to_plot[i].x_axis, self.lines_to_plot[i].means, '-o', label= Simulation.play[i], c=Simulation.avaiable_colours[i], alpha=0.3) # main_point
                self.ax.set_ylim(0,200)
                self.ax.errorbar(self.lines_to_plot[i].x_axis, self.lines_to_plot[i].means, yerr = self.lines_to_plot[i].stds, fmt = 'none' , ecolor= Simulation.avaiable_colours[i], alpha=0.3, capsize=2, capthick= 2)
            
            self.ax.legend()

    def graph_line_time(self,is_log = False, is_time = False):
        for i in range(len(self.lines_to_plot)):
            if is_time == True:
                if is_log == True:
                    lable = Simulation.play[i] + 'time to peak'
                    lable2 = Simulation.play[i] + ' simulation time (30)'
                    self.ax.semilogy(self.lines_to_plot[i].x_axis, self.lines_to_plot[i].mean_abs_pop_creation_keeper, '-x', label=lable, c=Simulation.avaiable_colours[i], alpha=0.3) # main_point
                    self.ax.semilogy(self.lines_to_plot[i].x_axis, self.lines_to_plot[i].abs_simulation_keeper, '-o', label= lable2, c=Simulation.avaiable_colours[i], alpha=0.3) # main_point
                else:
                    lable = Simulation.play[i] + 'time to peak'
                    lable2 = Simulation.play[i] + ' simulation time (30)'
                    self.ax.plot(self.lines_to_plot[i].x_axis, self.lines_to_plot[i].mean_abs_pop_creation_keeper, '-x', label= lable, c=Simulation.avaiable_colours[i], alpha=0.3) # main_point
                    self.ax.plot(self.lines_to_plot[i].x_axis, self.lines_to_plot[i].abs_simulation_keeper, '-o', label= lable2, c=Simulation.avaiable_colours[i], alpha=0.3) # main_point

            
            self.ax.legend()

    def draw_stencil_time(self):
        Simulation.fig = plt.figure()
        self.ax = Simulation.fig.add_subplot(111)
        self.ax.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        self.ax.set_xlim(0, 90)
        self.ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        self.ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        
        self.ax.set_xlabel('n')
        self.ax.set_ylabel('time to peak (s)')




class Line():
    def __init__(self):
        self.means = []
        self.stds = []
        self.n = 0
        self.x_axis = []
        self.mean_abs_pop_creation_keeper = []
        self.abs_simulation_keeper = []

    def generate_line_values(self, use):

        for n in range(10,self.n,10):
                iterations_taken = []
                number_of_time = 0
                time_start_for_simulation_to_finish = time.time()
                time_each_popolation = []
                while number_of_time < 30:
                    time_start_popolation_creation = time.time()
                    Pacific_island = Popolation(20,20,n)
                    Popolation.found = False
                    generation = 0
                    temp_number_of_time = number_of_time
                    if use == 'neural network':
                        clear_session()
                        Pacific_island.build_models(n)

                    while Popolation.found == False:
                        Pacific_island.calculate_every_individuals_fitness() 

                        Pacific_island.migration(1) 

                        if use == 'uniform crossover':
                            Pacific_island.uniform_cross_over()
                        elif use == 'one point crossover':
                            Pacific_island.cross_over_only()
                        elif use == 'mutation only':
                            Pacific_island.mutation_only()
                        elif use == 'randomised genetic map':
                            Individual.Random_genetic_map = True
                            Pacific_island.uniform_cross_over()
                        elif use == 'neural network':
                            Pacific_island.dense_generation(n)


                        print ('genome_lenght = ',n, 'test n = ', number_of_time, 'generation =', generation)
                        generation +=1

                        if use == 'one point crossover':
                            if generation == 400:
                                number_of_time -=1 # repeat the test
                                Popolation.found = True
                        else:
                            if n == 30:
                                if generation == 600:
                                    number_of_time -=1
                                    Popolation.found = True
                            elif n ==40:
                                if generation == 800:
                                    number_of_time -=1
                                    Popolation.found = True
                            elif n ==50:
                                if generation == 1300:
                                    number_of_time -=1
                                    Popolation.found = True
                            else:
                                if generation == 1400:
                                        number_of_time -=1
                                        Popolation.found = True

                    del Pacific_island
                    if temp_number_of_time == number_of_time:
                        iterations_taken.append(generation)
                    else:
                        pass
                    number_of_time += 1
                    time_finish_popolation_creation = time.time()
                    abs_pop_creation = time_finish_popolation_creation - time_start_popolation_creation
                    time_each_popolation.append(abs_pop_creation)
                print (iterations_taken)
                mean = sum(iterations_taken)/30
                std = np.std(iterations_taken)
                self.means.append(mean)
                self.stds.append(std)
                time_finish_for_simulation_to_finish = time.time()
                abs_simulation = time_finish_for_simulation_to_finish - time_start_for_simulation_to_finish
                mean_abs_pop_creation = sum(time_each_popolation)/30
                self.mean_abs_pop_creation_keeper.append(mean_abs_pop_creation)
                self.abs_simulation_keeper.append(abs_simulation)


test = 0

S = Simulation()

S.plot_graph()

S.show()

