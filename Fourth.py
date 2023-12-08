import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Disks:
    """A class describing many disks in a box"""

    def __init__(self, num_disks, radius=0.01, mass=1.0):
        """Create randomly several disks"""

        self.num_disks = num_disks
        self.radiuses = np.full(num_disks, radius)  # Convert radius to an array
        self.positions = rnd.random([num_disks, 2]) * (1 - 2 * radius) + radius
        self.velocities = rnd.random([num_disks, 2]) * 2 - 1.
        self.collision_events = np.zeros([num_disks, 4 + num_disks]) * np.nan
        self.walls = np.array([1., 1., 0., 0.])
        self.masses = np.full(num_disks, mass)

        self.radiuses[2] = 0.12
        self.masses[2] = 2.0

        for i in range(num_disks):
            self.update_collision_events(i)

    def apply_gravity(self, time_step):
        """Apply gravity to the velocities of all disks"""
        self.velocities += np.array([0, -9.8]) * time_step

    def update_collision_events(self, disk_index):
        """Update the collision events for the specified disk"""

        pos = self.positions[disk_index, :]
        vel = self.velocities[disk_index, :]
        walls = self.walls
        r = self.radiuses[disk_index]

        # no collision
        self.collision_events[disk_index, disk_index] = np.nan

        # if collisions with a wall
        for i in range(4):
            s = 1 if i < 2 else -1
            if s * vel[i % 2] >= 0:
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

    def step(self, time_step):
        """Make a time step with gravity"""

        # apply gravity
        self.apply_gravity(time_step)

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
            self.velocities[disk1_index, indices[1] % 2] = -self.velocities[disk1_index, indices[1] % 2]
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

        impulse = 2 * self.masses[disk1_index] * self.masses[disk2_index] * dot_product_vr / (sigma * (self.masses[disk1_index] + self.masses[disk2_index]))
        # change in momentum
        impulse_velocity = (impulse * delta_position / sigma)

        self.velocities[disk1_index, :] += impulse_velocity / self.masses[disk1_index]
        self.velocities[disk2_index, :] -= impulse_velocity / self.masses[disk2_index]
        self.update_collision_events(disk1_index)
        self.update_collision_events(disk2_index)

# Add an odd ball with higher size and mass
disks = Disks(3, 0.1, mass=1.5)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
line, = ax.plot([], [], 'go', ms=22)
oddball_line, = ax.plot([], [], 'bo', ms=32)  # Use blue color for the oddball
ax.vlines(0,0,1,'r','-')
ax.vlines(1,0,1,'r','-')
ax.hlines(0, 0,1,'r','-')
ax.hlines(1, 0,1,'r','-')
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)


def animate(i):
    disks.step(0.001)
    line.set_data(disks.positions[:, 0], disks.positions[:, 1])
    oddball_line.set_data(disks.positions[2, 0], disks.positions[2, 1])
    return line, oddball_line


anim = animation.FuncAnimation(fig, animate, interval=10, blit=False, save_count=500)

plt.show()

anim.save(r'C:\Users\dmcde\Desktop\PUT ANIM HERE\animation_oddball.gif', writer='pillow')