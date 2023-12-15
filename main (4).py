import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import time

start_time = time.time()
#Control Panel
N = 80           # Number of Disks, set here
Rad = 0.03       # Radius of Most Disks
Mass = 1         # Mass of Most Disks
Odd_Rad = 0.08   # Brazil Nut Radius
Odd_Mass = 2   # Brazil Nut Mass
g = 9.81       # Set to 0 to turn gravity off, set to any positive value to turn gravity on.
F = 3
A=25          # Used was 7

msgreen = 17     # For every 0.02 in radius us ms 11
msblue = 44

File_Path = r'C:\Users\dmcde\Desktop\PUT ANIM HERE'
File_Name = '\Test_1'

class Disks:
    """A class describing many disks in a box"""
    global_time_counter = 0  # Initialize a global time counter

    def __init__(self, num_disks, radius, mass, Odd_Rad, Odd_Mass):
        """Create randomly several disks"""

        self.num_disks = num_disks
        self.radiuses = np.full(num_disks, radius)  # Convert radius to an array
        self.positions = self.initialize_positions(num_disks, radius)
        self.velocities = rnd.random([num_disks, 2]) * 2 - 1.
        self.collision_events = np.zeros([num_disks, 4 + num_disks]) * np.nan
        self.walls = np.array([1., 1., 0., 0.])
        self.masses = np.full(num_disks, mass)

        self.radiuses[num_disks-1] = Odd_Rad
        self.masses[num_disks-1] = Odd_Mass

        for i in range(num_disks):
            self.update_collision_events(i)

    def initialize_positions(self, num_disks, radius):
        """Initialize positions without overlaps"""

        positions = rnd.random([num_disks, 2]) * (1 - 2 * radius) + radius

        # Set the position of the last disk (the "oddball")
        positions[num_disks - 1] = [0.5, 0 + Odd_Rad]

        # Adjust positions to eliminate overlaps
        for i in range(num_disks - 1):  # Exclude the "oddball"
            while np.any(np.sum((positions[i] - positions[:i]) ** 2, axis=1) < (2 * radius) ** 2) or \
                    np.sum((positions[i] - positions[num_disks - 1]) ** 2) < (radius + Odd_Rad) ** 2:  # Check for overlap with the "oddball"
                positions[i] = rnd.random(2) * (1 - 2 * radius) + radius

        return positions


    def update_collision_events(self, disk_index):
        """Update the collision events for the specified disk"""

        pos = self.positions[disk_index, :]
        vel = self.velocities[disk_index, :]
        #vel = vel + np.sqrt(vel**2 + 2*np.array([0, -9.8])*pos)
        walls = self.walls
        r = self.radiuses[disk_index]

        # no collision
        self.collision_events[disk_index, disk_index] = np.nan

        # if collisions with a wall
        for i in range(4):
            s = 1 if i < 2 else -1
            if s * vel[i % 2] > 0:
                self.collision_events[disk_index, i] = (walls[i] - pos[i % 2] - s * r) / vel[i % 2]
            else:
                self.collision_events[disk_index, i] = np.nan

        # if collisions with other disks
        for i in range(self.num_disks):
            if i != disk_index:
                collision_time = self.calculate_collision_time(disk_index, i)
                self.collision_events[disk_index, i + 4] = collision_time
                self.collision_events[i, disk_index + 4] = collision_time

    def calculate_collision_time(self, disk1_index, disk2_index):
        """Helper method to find disk-disk collision time"""

        radius1 = self.radiuses[disk1_index]
        radius2 = self.radiuses[disk2_index]

        sigma = radius1 + radius2  # sum of the radii for collision # effective diameter to see if they're in contact
        delta_velocity = self.velocities[disk2_index, :] - self.velocities[disk1_index, :]
        delta_position = self.positions[disk2_index, :] - self.positions[disk1_index, :]
        dot_product_vr = np.dot(delta_velocity, delta_position)

        if dot_product_vr > 0:
            return np.nan

        dot_product_vv = np.dot(delta_velocity, delta_velocity)
        discriminant = dot_product_vr ** 2 - dot_product_vv * (np.dot(delta_position, delta_position) - sigma ** 2)

        if discriminant < 0:
            return np.nan

        return -(dot_product_vr + np.sqrt(discriminant)) / dot_product_vv

    def step(self, time_step, frequency, amplitude, g, Odd_Mass, Mass, N):
        """Make a time step with gravity and shaking"""
        # Access the global time counter using the class reference
        Disks.global_time_counter += time_step
        # Calculate the current shaking force
        shaking_force = (np.sin(frequency * Disks.global_time_counter) * amplitude)
        print(shaking_force)
        # Apply gravity and shaking force
        self.velocities = self.velocities + np.array([0, (-g + (shaking_force / Mass))]) * time_step
        self.velocities[N - 1] = self.velocities[N - 1] + np.array([0, (-g + (shaking_force / Odd_Mass))]) * time_step
        for i in range(self.num_disks):
            self.update_collision_events(i)

        while time_step > 0:
            # find the event with the smallest time
            indices = np.unravel_index(np.nanargmin(self.collision_events), self.collision_events.shape)
            smallest_collision_time = self.collision_events[indices]

            if time_step < smallest_collision_time:
                self.positions += self.velocities * time_step
                self.collision_events -= time_step
                time_step = 0  # break out of the loop
            else:
                self.positions += self.velocities * smallest_collision_time
                self.collision_events -= smallest_collision_time
                self.handle_collision(indices)
                time_step -= smallest_collision_time

    def handle_collision(self, indices):
        """Handle collision event"""

        disk1_index = indices[0]  # disk1 = with the shortest collision time aka the one that experiences the collision

        # event if a collision with wall
        if indices[1] < 4:
            if indices[1] == 3:  # bottom wall
                self.positions[disk1_index, 1] = self.radiuses[disk1_index]  # place above the bottom wall
            restitution_coefficient = 0.6  # Adjust the coefficient of restitution as needed 0.65
            self.velocities[disk1_index, indices[1] % 2] = -restitution_coefficient * self.velocities[disk1_index, indices[1] % 2]
            self.update_collision_events(disk1_index)

        # event if collision with another disk
        else:
            disk2_index = indices[1] - 4  # in case of collision with another particle, disk2 = the second particle
            self.update_velocities_after_collision(disk1_index, disk2_index)

    def update_velocities_after_collision(self, disk1_index, disk2_index):
        """Update velocities after a disk-disk collision"""
        radius1 = self.radiuses[disk1_index]
        radius2 = self.radiuses[disk2_index]

        sigma = radius1 + radius2  # sum of the radii for collision
        delta_velocity = self.velocities[disk2_index, :] - self.velocities[disk1_index, :]
        delta_position = self.positions[disk2_index, :] - self.positions[disk1_index, :]
        dot_product_vr = np.dot(delta_velocity, delta_position)

        impulse = 2 * self.masses[disk1_index] * self.masses[disk2_index] * dot_product_vr / (
                sigma * (self.masses[disk1_index] + self.masses[disk2_index]))
        # change in momentum
        impulse_velocity = impulse * delta_position / sigma

        self.velocities[disk1_index, :] += impulse_velocity / self.masses[disk1_index]
        self.velocities[disk2_index, :] -= impulse_velocity / self.masses[disk2_index]
        self.update_collision_events(disk1_index)
        self.update_collision_events(disk2_index)


# Add an odd ball with higher size and mass
disks = Disks(N, Rad, Mass, Odd_Rad, Odd_Mass)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
line, = ax.plot([], [], 'go', ms=msgreen)
oddball_line, = ax.plot([], [], 'bo', ms=msblue)  # Use blue color for the oddball
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Creating lists to store positional values for the last ball
last_disk_x = []
last_disk_y = []

def animate(i):
    disks.step(0.05, F, A, g, Odd_Mass, Mass, N)
    line.set_data(disks.positions[:, 0], disks.positions[:, 1])
    oddball_line.set_data(disks.positions[N-1, 0], disks.positions[N-1, 1])

    # Appending positional values for the last disk to the lists
    last_disk_x.append(disks.positions[N-1, 0])
    last_disk_y.append(disks.positions[N-1, 1])

    return line, oddball_line

anim = animation.FuncAnimation(fig, animate, interval=75, blit=False, save_count=500)

#plt.show()

# Saving the animation as a GIF
anim.save(File_Path + File_Name + '.gif', writer='pillow')

# Creating the DataFrame for the last disk
df = pd.DataFrame({'LastDiskX': last_disk_x, 'LastDiskY': last_disk_y})

# Saving the DataFrame to a CSV file
df.to_csv(File_Path + File_Name + '.csv', index=False)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"The code took {elapsed_time} seconds to run.")
