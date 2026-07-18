import src.config as config
from src.physics.trajectory import Trajectory
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
        self.neighboringSatTrajectories = {}
        self.foreignTrajectory = None

        # Set up optimization in constructor to avoid re-compiling every step
        self.u = cp.Variable(self.N)
        self.x0_param = cp.Parameter(4)
        self.max_thrust_param = cp.Parameter(nonneg=True)

        self.leading_pos_param = cp.Parameter((self.N, 2))
        self.trailing_pos_param = cp.Parameter((self.N, 2))

        #  Get physics
        A, B = orbit.eci_to_cw_matrix(x0, v0)
        Ad, Bd = orbit.discretize_CW(A, B, self.dt)
        self.Ad = Ad
        self.Bd = Bd
        Bd_flat = Bd.flatten()

        # State rollout
        x_expr = [self.x0_param]
        for k in range(self.N):
            x_expr.append(Ad @ x_expr[k] + Bd_flat * self.u[k])

        # Cost function (sum of squares formulation for DPP compliance)
        Q = config.MPC_Q_MATRIX
        q_diag = np.diag(Q)
        R = config.MPC_R_MATRIX
        cost = 0.0
        for k in range(self.N):
            cost += (
                q_diag[0] * cp.square(x_expr[k][0]) +
                q_diag[1] * cp.square(x_expr[k][1]) +
                q_diag[2] * cp.square(x_expr[k][2]) +
                q_diag[3] * cp.square(x_expr[k][3]) +
                R * cp.square(self.u[k])
            )
        cost += (
            q_diag[0] * cp.square(x_expr[self.N][0]) +
            q_diag[1] * cp.square(x_expr[self.N][1]) +
            q_diag[2] * cp.square(x_expr[self.N][2]) +
            q_diag[3] * cp.square(x_expr[self.N][3])
        )

        # Constraints
        constraints = [
            self.u >= -self.max_thrust_param,
            self.u <= self.max_thrust_param
        ]

        phi_max = config.CROSSLINK_GIMBAL_RANGE
        r_orbit = np.linalg.norm(x0)
        max_along_track = 2 * r_orbit * np.sin(phi_max)
        min_along_track = config.SAFETY_DISTANCE

        # Add crosslink constraints to the pre-compiled problem
        for k in range(1, self.N + 1):
            idx = k - 1
            # Leading neighbor constraints
            dx_lead = self.leading_pos_param[idx, 0] - x_expr[k][0]
            dy_lead = self.leading_pos_param[idx, 1] - x_expr[k][1]
            constraints.append(dy_lead * np.sin(phi_max) >= dx_lead * np.cos(phi_max))
            constraints.append(dy_lead * np.sin(phi_max) >= -dx_lead * np.cos(phi_max))
            constraints.append(dy_lead <= max_along_track)
            constraints.append(dy_lead >= min_along_track)

            # Trailing neighbor constraints
            dx_trail = self.trailing_pos_param[idx, 0] - x_expr[k][0]
            dy_trail = self.trailing_pos_param[idx, 1] - x_expr[k][1]
            constraints.append((-dy_trail) * np.sin(phi_max) >= dx_trail * np.cos(phi_max))
            constraints.append((-dy_trail) * np.sin(phi_max) >= -dx_trail * np.cos(phi_max))
            constraints.append(-dy_trail <= max_along_track)
            constraints.append(-dy_trail >= min_along_track)

        # Add equal spacing goal
        spacing_weight = config.MPC_W_MATRIX
        for k in range(1, self.N + 1):
            idx = k -1
            lead_spacing = self.leading_pos_param[idx, 1] - x_expr[k][1]
            trail_spacing = x_expr[k][1] - self.trailing_pos_param[idx, 1]
            # minimize square difference for sign
            cost += spacing_weight * cp.square(lead_spacing - trail_spacing)


        self.prob = cp.Problem(cp.Minimize(cost), constraints)
        self.x_expr = x_expr


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
            # Create difference state vector in LVLH frame
            pos_lvlh, vel_lvlh = orbit.eci_to_lvlh(ref_pos_eci, ref_vel_eci, pos, vel)                                                                                                                                                                                                                            
            x0 = np.array([pos_lvlh[0], pos_lvlh[1], vel_lvlh[0], vel_lvlh[1]])                                                                              

            # Set parameters for pre-compiled problem
            self.x0_param.value = x0
            self.max_thrust_param.value = max_thrust

            # Default: fill neighbor parameters with their nominal relative positions
            r_orbit = np.linalg.norm(ref_pos_eci)
            nominal_spacing_dist = 2 * r_orbit * np.sin(np.deg2rad(10.0))
            
            leading_val = np.zeros((self.N, 2))
            trailing_val = np.zeros((self.N, 2))
            
            leading_val[:, 1] = nominal_spacing_dist
            trailing_val[:, 1] = -nominal_spacing_dist
            
            # Update with actual neighbor trajectories if available
            for neighborID, neighborTraj in self.neighboringSatTrajectories.items():
                if neighborTraj is not None:
                    neighbor_pos_0_eci, neighbor_vel_0_eci = neighborTraj.get_state_at_time(time)
                    neighbor_pos_0_lvlh, _ = orbit.eci_to_lvlh(ref_pos_eci, ref_vel_eci, neighbor_pos_0_eci, neighbor_vel_0_eci)
                    
                    isLeading = neighbor_pos_0_lvlh[1] > 0
                    
                    for k in range(1, self.N + 1):
                        idx = k - 1
                        next_time = time + k*self.dt
                        ref_k_eci, v_ref_k_eci = self.get_nominal_state(next_time)
                        neighbor_k_eci, v_neighbor_k_eci = neighborTraj.get_state_at_time(next_time)
                        neighbor_k_lvlh, _ = orbit.eci_to_lvlh(ref_k_eci, v_ref_k_eci, neighbor_k_eci, v_neighbor_k_eci)
                        
                        if isLeading:
                            leading_val[idx, 0] = neighbor_k_lvlh[0]
                            leading_val[idx, 1] = neighbor_k_lvlh[1]
                        else:
                            trailing_val[idx, 0] = neighbor_k_lvlh[0]
                            trailing_val[idx, 1] = neighbor_k_lvlh[1]
                            
            self.leading_pos_param.value = leading_val
            self.trailing_pos_param.value = trailing_val
            
            # Solve pre-compiled QP
            self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

            success = self.prob.status in ["optimal", "optimal_inaccurate"]
            if success:                                                                                             
                self.stored_controls = self.u.value                                                                                                                          
            else:                                                                                                                                            
                self.stored_controls = np.zeros(self.N) # Fallback to zero thrust on failure 
                
            # Unroll and save the trajectory plan
            self.computedTrajectory = self.unroll_trajectory(time, self.x_expr, success)
            self.last_mpc_run_time = time

            # Original non pre compiles
            # A, B = orbit.eci_to_cw_matrix(ref_pos_eci, ref_vel_eci)
            # Ad, Bd = orbit.discretize_CW(A, B, self.dt)
            # Bd_flat = Bd.flatten()     
            # u = cp.Variable(self.N)
            # x = [x0]
            # for k in range(self.N):
            #     x.append(Ad @ x[k] + Bd_flat * u[k])   
            # Q = config.MPC_Q_MATRIX
            # R = config.MPC_R_MATRIX
            # cost = 0.0                                                                                                                                       
            # for k in range(self.N):
            #     state_cost = cp.quad_form(x[k], Q)
            #     fuel_cost = R * cp.square(u[k])
            #     cost += state_cost + fuel_cost
            # cost += cp.quad_form(x[self.N], Q)           
            # constraints = [                                                                                                                                  
            #     u >= -max_thrust,                                                                                                                            
            #     u <= max_thrust                                                                                                                              
            # ]
            # for neighborID, neighborTraj in self.neighboringSatTrajectories.items():
            #     if neighborTraj is not None:
            #         neighbor_pos_0_eci, neighbor_vel_0_eci= neighborTraj.get_state_at_time(time)
            #         neighbor_pos_0_lvlh, neighbor_vel_0_lvlh = orbit.eci_to_lvlh(ref_pos_eci, ref_vel_eci, neighbor_pos_0_eci, neighbor_vel_0_eci)
            #         isLeading = neighbor_pos_0_lvlh[1] > 0
            #         phi_max = config.CROSSLINK_GIMBAL_RANGE                                                                            
            #         r_orbit = np.linalg.norm(ref_pos_eci)                                                                                                  
            #         max_along_track = 2 * r_orbit * np.sin(phi_max)
            #         min_along_track = config.SAFETY_DISTANCE
            #         for k in range(4, self.N + 1, 4):
            #             next_time = time + k*self.dt
            #             ref_k_eci, v_ref_k_eci = self.get_nominal_state(next_time)
            #             neighbor_k_eci, v_neighbor_k_eci = neighborTraj.get_state_at_time(next_time)
            #             neighbor_k_lvlh, v_neighbor_k_lvlh = orbit.eci_to_lvlh(ref_k_eci, v_ref_k_eci, neighbor_k_eci, v_neighbor_k_eci)
            #             x_rel = neighbor_k_lvlh[0] - x[k][0]
            #             y_rel = neighbor_k_lvlh[1] - x[k][1]
            #             if isLeading:
            #                 constraints.append(y_rel * np.sin(phi_max) >= cp.abs(x_rel) * np.cos(phi_max))
            #                 constraints.append(y_rel <= max_along_track)
            #                 constraints.append(y_rel >= min_along_track)
            #             else:
            #                 constraints.append(-y_rel * np.sin(phi_max) >= cp.abs(x_rel) * np.cos(phi_max))
            #                 constraints.append(-y_rel <= max_along_track)
            #                 constraints.append(-y_rel >= min_along_track)
            # prob = cp.Problem(cp.Minimize(cost), constraints)   
            # prob.solve(solver=cp.OSQP, verbose=False)      
            
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

        xk_val = None
        for k in range(self.N + 1):
            t_future = time + k * self.dt
            pos_nom_k, vel_nom_k = self.get_nominal_state(t_future)
            
            if success:
                xk_val = x_expressions[k].value
            else:
                # Fallback: propagate passively using Ad under zero thrust (u=0)
                if k == 0:
                    xk_val = self.x0_param.value
                else:
                    xk_val = self.Ad @ xk_val
            
            pos_lvlh = np.array([xk_val[0], xk_val[1], 0.0])
            vel_lvlh = np.array([xk_val[2], xk_val[3], 0.0])
            
            r_unit = pos_nom_k / np.linalg.norm(pos_nom_k)
            h_unit = np.cross(pos_nom_k, vel_nom_k)
            h_unit /= np.linalg.norm(h_unit)
            theta_unit = np.cross(h_unit, r_unit)
            R_lvlh_to_eci = np.vstack([r_unit, theta_unit, h_unit]).T
            
            pos_eci = pos_nom_k + R_lvlh_to_eci @ pos_lvlh
            vel_eci = vel_nom_k + R_lvlh_to_eci @ vel_lvlh

            pred_pos.append(pos_eci)
            pred_vel.append(vel_eci)
            pred_times.append(t_future)

        return Trajectory(
            ECI_position_in=np.array(pred_pos),
            ECI_velocity_in=np.array(pred_vel),
            time_in=np.array(pred_times)
        )      