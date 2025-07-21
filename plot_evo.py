import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

#########################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot_params")
    parser.add_argument('graphs', nargs='+', help='graphs to plot')
    parser.add_argument("--interpolation", type=int, help="number of points to interpolate")

    args = parser.parse_args()

    os.chdir("./Plots")

    plt.figure(figsize=(20,5))
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, g in enumerate(args.graphs):

        color = colors[i % len(colors)]
        to_draw = np.load(g)
    
        indices = [item[0] for item in to_draw]
        valeurs = [item[1] for item in to_draw]
        
        indices_smooth = np.linspace(min(indices), max(indices), args.interpolation)  # Indices interpolés
        valeurs_smooth = np.interp(indices_smooth, indices, valeurs)  # Interpolation des valeurs
        
        plt.plot(indices_smooth, valeurs_smooth, label=g, color=color)
        plt.axhline(y=np.mean(valeurs), linestyle='--', color=color, label=f'Mean: {np.mean(valeurs):.2f} (Std: {np.std(valeurs):.2f})')



    plt.xlabel('Time')
    plt.ylabel('Quantity')
    # plt.title('')
    plt.legend()
    plt.grid(True)
    plt.show()

    os.chdir("../")
    print("Fin du script. Tout s’est bien déroulé.")