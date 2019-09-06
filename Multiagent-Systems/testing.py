import numpy as np

class DroneGroup:
    def __init__(self):
        self.drones = []
        for i in range(0, 5):
            self.drones.append(Drone())
    def request_to_all(self, task, params):
        for drone in self.drones:
            drone.task(params)

class Drone:
    def __init__(self):
        self.available = True
        self.pos = np.random.rand(2)
        self.distances_trush = np.array([])

    def set_trush_distances(self):
        self.distances_trush = np.zeros((len(trushes)))
        for i in range(len(trushes)):
            self.distances_trush[i] = np.linalg.norm(trushes[i].pos- self.pos)

    def petition_grab_trush(self, trush, distance):
        if not self.available:
            return True
        else:
            if np.linspace.norm(self.pos-trush.pos) < distance:


class Trush:
    def __init__(self):
        self.available = True
        self.pos = np.random.rand(2)
        self.target = np.random.rand(2)


drones = []
trushes = []



for i in range(0,3):
    trushes.append(Trush())


for drone in drones:
    drone.look_for_trush()

for drone in drones:
    drone.look_for_trush()
