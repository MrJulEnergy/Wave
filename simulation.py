import numpy as np
import pickle
import tqdm

class Variables:
    # Discretisation
    dx = 1e-2
    dt = 1e-4

    # Boundaries
    boundary = (0, 2) # Square Lattice Grid
    T = 30

    # Number of Steps
    n_t = int(T/dt)
    n_x = int((boundary[1]-boundary[0])/dx)

    # Physics
    c = 0.1 # speed of light
    
    # 2nd Derivative
    C = np.power(c/dx, 2)

    # Animation
    fps = 30
    stride_length = round(1/(dt * fps))

    # Extern
    dumpfile = "dump.save"


class Simulation:
    def __init__(self):
        # Define Grids of Simulation Area
        self.E = np.zeros(shape=(Variables.n_x, Variables.n_x)) # Electric Field
        self.flow = np.zeros_like(self.E) # Flow of Electric field dE/dt

        # Set Inital Conditions
        #self.E[Variables.n_x//2, int(Variables.n_x//2.2)] = 1
        #self.E[Variables.n_x//2, int(Variables.n_x - Variables.n_x//2.2)] = 1

    def laplace_E(self):
        derivative = np.zeros_like(self.E)

        left = np.roll(self.E, -1, axis=0) # f(x-dx)
        right = np.roll(self.E, 1, axis=0) # f(x+dx)
        up = np.roll(self.E, -1, axis=1) # f(y-dx)
        down = np.roll(self.E, 1, axis=1) # f(y+dx)

        derivative = Variables.C * (left + right + down + up - 4*self.E)
        return derivative

    def apply_boundary_conditions(self):
        self.E[:, 0] = 0
        self.E[:, -1] = 0
        self.E[0, :] = 0
        self.E[-1, :] = 0

        self.E[0, :] = 0.02*np.sin(2*np.pi * self.t*Variables.dt * 1.5)
        self.E[50:51, 0:90] = 0
        self.E[50:51, 95:105] = 0
        self.E[50:51, 110:-1] = 0

    def symplectic_euler_step(self):
        self.flow = self.flow + Variables.dt * self.laplace_E()
        self.E = self.E + Variables.dt * self.flow
        self.apply_boundary_conditions()

    def run(self):
        file = open(Variables.dumpfile, "wb")
        pickle.dump((self.E, 0), file)

        for self.t in tqdm.tqdm(range(1, Variables.n_t)):
            self.symplectic_euler_step()
            if self.t % Variables.stride_length == 0:
                pickle.dump((self.E, self.t*Variables.dt), file)
        
        file.close()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()
