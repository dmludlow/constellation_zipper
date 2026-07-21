"""
Object used to configure the intersection between a rocket launch and an orbital ring
"""

from src.physics.trajectory import Trajectory
from src.physics import orbit
import src.simulation.config as config

import numpy as np

class Rocket:

    def __init__(self, GTO_burn_time: float):
        """
        Initializes a Rocket object with the specified intersection time and altitude.

        GTO Perigee: 200 km

        Parameters:
        INTERSECT_TIME_S (float): The time at which the rocket intersects the orbital ring (in seconds).
        """
        # Simulate the change from a lower holding orbit to a GTO

        # Generate data for 4 hours before and after GTO brun
        time_vector = np.arange(0, config.SIMULATION_DURATION_S, config.SIMULATION_TIME_STEP_S)

        # Initial LEO holding orbit
        LEO_pos_eci, LEO_vel_eci = orbit.generate_orbit(altitude=200000, degrees_long=0.0)
        burned = False

        # Propogate orbit
        sum_pos_ECI = []
        sum_vel_ECI = []

        curr_pos = LEO_pos_eci
        curr_vel = LEO_vel_eci
        for t in time_vector:
            if t >= (GTO_burn_time) and not burned:
                # At GTO burn, assume perfectly impulsive burn to GTO orbit
                pos, vel = self.burn_GTO(curr_pos, curr_vel)
                burned = True
            else: 
                # Propogate orbit
                pos, vel = orbit.propogate_orbit_rk4(curr_pos, 
                                                     curr_vel, 
                                                     mass=1,        # irrelevant mass, avoiding division by zero
                                                     commandedForce = np.zeros(3),    # no thrust
                                                     dt = config.SIMULATION_TIME_STEP_S)

            curr_pos = pos
            curr_vel = vel

            sum_pos_ECI.append(pos)
            sum_vel_ECI.append(vel)
 
        
        self.trajectory = Trajectory(sum_pos_ECI, sum_vel_ECI, time_vector)


    def burn_GTO(self, last_pos_eci, last_vel_eci) -> tuple[np.ndarray, np.ndarray]:
        """
        Burns into a new GTO orbit
        Returns the new position and velocity after burn
        """
        apogee_altitude_m = 35786000  # meters
        perigee_altitude_m = np.linalg.norm(last_pos_eci)  # meters

        # determine new velocity for GTO orbit at perigee
        mu = config.EARTH_GRAVITATIONAL_PARAMETER
        r_p = perigee_altitude_m
        r_a = config.EARTH_EQUATORIAL_RADIUS_M + apogee_altitude_m
        v_gto = np.sqrt((2.0 * mu * r_a) / (r_p * (r_p + r_a)))
        
        # scale and repoint velocity vector
        vel_unit = last_vel_eci / np.linalg.norm(last_vel_eci)
        new_vel_eci = v_gto * vel_unit

        # step last position with new velocity by 1 step to get new position
        new_pos, new_vel = orbit.propogate_orbit_rk4(last_pos_eci, 
                                                     new_vel_eci, 
                                                     mass=1,        # irrelevant mass, avoiding division by zero
                                                     commandedForce = np.zeros(3),    # no thrust
                                                     dt = config.SIMULATION_TIME_STEP_S)

        return new_pos, new_vel
        

    # in python you can usually just index into the trajectory object directly