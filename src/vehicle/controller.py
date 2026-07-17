import src.config as config
from src.vehicle.trajectory import Trajectory
import src.physics.orbit as orbit
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.vehicle.satellite import Satellite

import numpy as np
import cvxpy as cp

class Controller:
    """
    Implements a distributed model predictive control algorithm to open a slot in a satellite constellation
    """

    # MPC parameters
    N: int = config.MPC_HORIZON_LENGTH                             # MPC horizon length
    dt: float = config.MPC_TIME_STEP                               # Time step for the MPC in seconds
    safety_distance: float = config.SAFETY_DISTANCE                # Minimum distance to maintain from other satellites in meters

    # MPC results
    computedControl: np.ndarray = np.zeros(3)                      # Computed control inputs
    computedTrajectory: Trajectory = None                          # Computed trajectory of the satellite

    # Inputs
    neighboringSatTrajectories: dict[int, Trajectory] = {}         # Dictionary to hold neighboring satellite trajectories for MPC
    foreignTrajectory: Trajectory                                   # Trajectory of object ot be avoided

    # Initial position and velocity (to propogate nominal reference position)
    initialPosition: np.ndarray = np.zeros(3)                       # Initial position of the satellite in ECI frame
    initialVelocity: np.ndarray = np.zeros(3)                       # Initial velocity of the satellite

    time: float = 0.0                                                # Current time in seconds


    def __init__(self, x0: np.ndarray, v0: np.ndarray):
        """
        Initializes the controller with the initial state of the satellite.
        """
        self.initialPosition = x0
        self.initialVelocity = v0
        self.last_mpc_run_time = -9999.0  # Time of last MPC optimization execution
        self.stored_controls = None       # Array to cache calculated optimal control inputs

        # Set up optimization in constructor to avoid re-compiling every step


    def get_nominal_state(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the nominal circular ECI position and velocity vectors at a specific time.
        """
        r0 = self.initialPosition
        v0 = self.initialVelocity
        
        r_mag = np.linalg.norm(r0)
        n = np.sqrt(config.EARTH_GRAVITATIONAL_PARAMETER / (r_mag**3))
        
        # Analytical circular propagation
        pos_nom = np.cos(n * time) * r0 + (np.sin(n * time) / n) * v0
        vel_nom = -n * np.sin(n * time) * r0 + np.cos(n * time) * v0
        
        return pos_nom, vel_nom


    def receive_neighboring_trajectories(self, neighborID: int, neighborTraj: Trajectory):
        """
        Receives the predicted trajectory of a neighboring satellite
        """
        self.neighboringSatTrajectories[neighborID] = neighborTraj


    def receive_foreign_trajectory(self, foreignTraj: Trajectory, safety_distance: Optional[float] = None):
        """
        Receives the predicted trajectory of a foreign object to be avoided
        Optional: safety distance can be specified to override the default safety distance for this object
        """
        self.foreignTrajectory = foreignTraj
        if safety_distance is not None:
            self.safety_distance = safety_distance


    def propogate_neighboring_trajectories(self):
        """
        In case of lost conenction, propogates the neighboring satellite trajectories
        Assume neighboring sat has equal mass
        """
        pass
            

    def solve_mpc(self, time: float, pos: np.ndarray, vel: np.ndarray, mass: float, max_thrust: float) -> np.ndarray:
        """
        Solves the MPC optimization problem using the local state and parameters.
        Returns the computed control force in the ECI frame.
        """

        self.time = time

        # Get current nominal state 
        ref_pos_eci, ref_vel_eci = self.get_nominal_state(time)

        # Check if it is time to re-solve the optimization (every self.dt seconds)
        time_since_last_run = time - self.last_mpc_run_time
        
        if time_since_last_run >= self.dt or self.stored_controls is None:
            # Create LVLH frame and physics
            A, B = orbit.eci_to_cw_matrix(ref_pos_eci, ref_vel_eci)
            Ad, Bd = orbit.discretize_CW(A, B, self.dt)
            
            # Create difference state vector in LVLH frame
            pos_lvlh, vel_lvlh = orbit.eci_to_lvlh(ref_pos_eci, ref_vel_eci, pos, vel)                                                                                                                                                                                                                            
            x0 = np.array([pos_lvlh[0], pos_lvlh[1], vel_lvlh[0], vel_lvlh[1]])                                                                              
                                                                                                                                                             
            # Flatten Bd (which is 4x1) for simple 1D vector math                                                                                            
            Bd_flat = Bd.flatten()     

            # Build the optimization problem using cvxpy
            u = cp.Variable(self.N) # decision variable

            # State rollout                                                                                          
            x = [x0]                                                                                                                                         
            for k in range(self.N):                                                                                                                          
                x.append(Ad @ x[k] + Bd_flat * u[k])   

            # Cost function
            Q = config.MPC_Q_MATRIX
            R = config.MPC_R_MATRIX

            cost = 0.0                                                                                                                                       
            for k in range(self.N):                                                                                                                          
                # 1. State deviation cost
                state_cost = cp.quad_form(x[k], Q)
                
                # 2. Control cost
                fuel_cost = R * cp.square(u[k])
                
                cost += state_cost + fuel_cost
                
            # Terminal cost 
            cost += cp.quad_form(x[self.N], Q)           

            # Constraints
            constraints = [                                                                                                                                  
                u >= -max_thrust,                                                                                                                            
                u <= max_thrust                                                                                                                              
            ]

            #TODO
            # Avoidance constraints for incoming foreign object
            

            # Crosslink constraints
            for neighborID, neighborTraj in self.neighboringSatTrajectories.items():
                if neighborTraj is not None:
                    neighbor_pos_0_eci, neighbor_vel_0_eci= neighborTraj.get_state_at_time(time)
                    neighbor_pos_0_lvlh, neighbor_vel_0_lvlh = orbit.eci_to_lvlh(ref_pos_eci, ref_vel_eci, neighbor_pos_0_eci, neighbor_vel_0_eci)

                    isLeading = neighbor_pos_0_lvlh[1] > 0

                    phi_max = config.CROSSLINK_GIMBAL_RANGE                                                                            
                    r_orbit = np.linalg.norm(ref_pos_eci)                                                                                                  
                    max_along_track = 2 * r_orbit * np.sin(phi_max)   # Max distance before chord angle > phi max                                          
                    min_along_track = config.SAFETY_DISTANCE          # Prevent collision  

                    for k in range(4, self.N + 1, 4):
                        next_time = time + k*self.dt
                        ref_k_eci, v_ref_k_eci = self.get_nominal_state(next_time)
                        neighbor_k_eci, v_neighbor_k_eci = neighborTraj.get_state_at_time(next_time)
                        neighbor_k_lvlh, v_neighbor_k_lvlh = orbit.eci_to_lvlh(ref_k_eci, v_ref_k_eci, neighbor_k_eci, v_neighbor_k_eci)

                        # Relative coordinates
                        x_rel = neighbor_k_lvlh[0] - x[k][0]
                        y_rel = neighbor_k_lvlh[1] - x[k][1]

                        # Save to constraints
                        if isLeading:
                            constraints.append(y_rel * np.sin(phi_max) >= cp.abs(x_rel) * np.cos(phi_max))
                            constraints.append(y_rel <= max_along_track)  # Leading neighbor must be within gimbal range
                            constraints.append(y_rel >= min_along_track)  # Leading neighbor must be at least safety distance away
                        else:
                            constraints.append(-y_rel * np.sin(phi_max) >= cp.abs(x_rel) * np.cos(phi_max))
                            constraints.append(-y_rel <= max_along_track) # Trailing neighbor must be within gimbal range
                            constraints.append(-y_rel >= min_along_track) # Trailing neighbor must be at least safety distance away

            # Solve QP
            prob = cp.Problem(cp.Minimize(cost), constraints)   
            prob.solve(solver=cp.OSQP, verbose=False)      

            success = prob.status in ["optimal", "optimal_inaccurate"]
            if success:                                                                                             
                self.stored_controls = u.value                                                                                                                          
            else:                                                                                                                                            
                self.stored_controls = np.zeros(self.N) # Fallback to zero thrust on failure 
                
            # Unroll and save the trajectory plan
            self.computedTrajectory = self.unroll_trajectory(time, x, success)
            self.last_mpc_run_time = time
            
        # Control input for the current simulation step
        idx = int((time - self.last_mpc_run_time) // self.dt)
        idx = min(idx, self.N - 1)
        current_u = self.stored_controls[idx]
        
        # Convert along-track force to ECI frame using current nominal frame orientation
        along_track_force_lvlh = np.array([0.0, current_u, 0.0])                                                                                   
        self.computedControl = orbit.lvlh_to_eci_force(ref_pos_eci, ref_vel_eci, along_track_force_lvlh)

        return self.computedControl


    def unroll_trajectory(self, time: float, x_expressions: list, success: bool) -> Trajectory:
        """
        Converts the relative state expressions from the MPC solver
        into an ECI Trajectory object.
        """
        pred_pos = []
        pred_vel = []
        pred_times = []

        for k in range(self.N + 1):
            t_future = time + k * self.dt
            pos_nom_k, vel_nom_k = self.get_nominal_state(t_future)
            
            if success:
                xk_val = x_expressions[k] if k == 0 else x_expressions[k].value
                pos_lvlh = np.array([xk_val[0], xk_val[1], 0.0])
                vel_lvlh = np.array([xk_val[2], xk_val[3], 0.0])
                
                r_unit = pos_nom_k / np.linalg.norm(pos_nom_k)
                h_unit = np.cross(pos_nom_k, vel_nom_k)
                h_unit /= np.linalg.norm(h_unit)
                theta_unit = np.cross(h_unit, r_unit)
                R_lvlh_to_eci = np.vstack([r_unit, theta_unit, h_unit]).T
                
                pos_eci = pos_nom_k + R_lvlh_to_eci @ pos_lvlh
                vel_eci = vel_nom_k + R_lvlh_to_eci @ vel_lvlh
            else:
                pos_eci = pos_nom_k
                vel_eci = vel_nom_k

            pred_pos.append(pos_eci)
            pred_vel.append(vel_eci)
            pred_times.append(t_future)

        return Trajectory(
            ECI_position_in=np.array(pred_pos),
            ECI_velocity_in=np.array(pred_vel),
            time_in=np.array(pred_times)
        )      