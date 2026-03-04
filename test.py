import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Load your new arm
# useFixedBase=True ensures the base cylinder doesn't fall over
robot_id = p.loadURDF("sheesh.urdf", [0, 0, 0], useFixedBase=False)

while True:
    p.stepSimulation()
    time.sleep(1./240.)