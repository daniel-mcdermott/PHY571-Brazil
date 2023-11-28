import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Disks:
    """A class describing many disks in a box"""

    def __init__(self, num_disks, radius=0.01):
        """Create randomly several disks"""

        self.num_disks = num_disks
        self.radius = radius
        self.positions = rnd.random([num_disks, 2]) * (1 - 2 * radius) + radius
        self.velocities = rnd.random([num_disks, 2]) * 2 - 1.
        self.collision_events = np.zeros([num_disks, 4 + num_disks]) * np.nan
        self.walls = np.array([1., 1., 0., 0.])
        self.mass = mass
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
        r = self.radius

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

        sigma = 2 * self.radius  # effective diameter to see if they're in contact
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

        # Apply gravity
        self.apply_gravity(time_step)

        while time_step > 0:
            # Find event with smallest time
            indices = np.unravel_index(np.nanargmin(self.collision_events), self.collision_events.shape)
            smallest_collision_time = self.collision_events[indices]

            if time_step < smallest_collision_time:
                self.positions += self.velocities * time_step
                self.collision_events -= time_step
                time_step = 0  # Break out of the loop
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

        sigma = 2 * self.radius
        delta_velocity = self.velocities[disk2_index, :] - self.velocities[disk1_index, :]
        delta_position = self.positions[disk2_index, :] - self.positions[disk1_index, :]
        dot_product_vr = np.dot(delta_velocity, delta_position)

        impulse = 2 * 1 * 1 * dot_product_vr / (sigma * (1 + 1))  # change in momentum
        impulse_velocity = impulse * delta_position / sigma

        self.velocities[disk1_index, :] += impulse_velocity
        self.velocities[disk2_index, :] -= impulse_velocity
        self.update_collision_events(disk1_index)
        self.update_collision_events(disk2_index)


disks = Disks(15, 0.08)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
line, = ax.plot([], [], 'ro', ms=48)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)


def animate(i):
    disks.step(0.05)
    line.set_data(disks.positions[:, 0], disks.positions[:, 1])
    plt.show()
    return line,


anim = animation.FuncAnimation(fig, animate, interval=10, blit=False, save_count=500)

anim.save('animation_gravity.gif', writer='pillow')
