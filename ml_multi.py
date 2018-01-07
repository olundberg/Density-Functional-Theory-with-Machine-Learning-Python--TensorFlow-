from ase import Atom, Atoms
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase import io
from ase.db import connect

from amp.utilities import randomize_images
from amp import Amp
from amp.descriptor import *
from amp.regression import *
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction
from ase.visualize import view
import os
from glob import glob

###############################################################################

def test():
    # LOAD DATA
    path = "/home/unknown/Dropbox/Oscar/Skola/LTU/Project course engineering physics/Results/CNT Capped/data" # Path to data
    results = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.db'))]

    print("Files found and imported:")
    for i in results:
    	print i

    images = []
    for i in results:
        images.extend(io.read(i,index=':'))



    #images = images[:500]
    print('Total number of images imported:  %i' % len(images))

    # Train the calculator.
    #train_images, test_images = randomize_images(images)
    train_images = images
    #calc = Amp(descriptor=Gaussian(),
    #          model=NeuralNetwork(hiddenlayers=(20, 20)))
    calc = Amp.load('parameters-checkpoint-600.amp')
    #calc.model.lossfunction = LossFunction(
    #    convergence={'energy_rmse': 1,
    #                'force_rmse': 0.01})
    #calc.train(train_images)

    # Plot and test the predictions.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot

    fig, ax = pyplot.subplots()
    #train = open('train.csv','a')
    #test = open('test.csv','a')
	
    #for image in train_images:
    #    actual_energy = image.get_potential_energy()
    #    predicted_energy = calc.get_potential_energy(image)
    #    ax.plot(actual_energy, predicted_energy, 'b.')
    #    string = "%s,%s\n" %(actual_energy,predicted_energy)
    #    train.write(string)

    #train.close()

    for image in train_images:
        actual_energy = image.get_potential_energy()
        predicted_energy = calc.get_potential_energy(image)
        ax.plot(actual_energy, predicted_energy, 'r.')
    	string = "%s,%s\n" %(actual_energy,predicted_energy)
    	
	    


    ax.set_xlabel('Actual energy, eV')
    ax.set_ylabel('Amp energy, eV')

    fig.savefig('parityplot.png')

###############################################################################

if __name__ == '__main__':
    test()
