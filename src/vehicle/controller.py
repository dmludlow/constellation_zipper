import src.simulation.config as config
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

    # MPC parameters shared by all controller instances
    N: int = config.MPC_HORIZON_LENGTH_S                             # MPC horizon length
    dt: float = config.MPC_TIME_STEP_S                               # Time step for the MPC in seconds
    safety_distance: float = config.SAFETY_DISTANCE_M                # Minimum distance to maintain from other satellites in meters

    # Controller specific
    initialPosition: np.ndarray
    initialVelocity: np.ndarray
    time: float
    computedControl: np.ndarray
    computedTrajectory: Optional[Trajectory]
    neighboringSatTrajectories: dict[int, Trajectory]
    foreignTrajectory: Optional[Trajectory]

    def __init__(self, x0: np.ndarray, v0: np.ndarray):
        """
        Initializes the controller with the initial state of the satellite.
        """
        self.initialPosition = x0
        self.initialVelocity = v0
        self.time = 0.0
        self.last_mpc_run_time = -9999.0  # Time of last MPC optimization execution
        self.stored_controls = None       # Array to cache calculated optimal control inputs
        
        # Prevent shared class-level mutable defaults
        self.computedControl = np.zeros(3)
        self.computedTrajectory = None
        self.neighboringSatTrajectories = {}
        self.foreignTrajectory = None

        # Set up optimization in constructor to avoid re-compiling every step
        self.u = cp.Variable(self.N)
        self.x0_param = cp.Parameter(4)
        self.max_thrust_param = cp.Parameter(nonneg=True)

        self.leading_pos_param = cp.Parameter((self.N, 2))
        self.trailing_pos_param = cp.Parameter((self.N, 2))

        # Parameters for rocket avoidance tangent wall (Lists of parameters to maintain DPP compliance)
        self.foreign_n_param_x = [cp.Parameter() for _ in range(self.N)]
        self.foreign_n_param_y = [cp.Parameter() for _ in range(self.N)]
        self.foreign_dist_param = [cp.Parameter() for _ in range(self.N)]

        #  Get physics
        A, B = orbit.eci_to_cw_matrix(x0, v0)
        Ad, Bd = orbit.discretize_CW(A, B, self.dt)
        self.Ad = Ad
        self.Bd = Bd
        Bd_flat = Bd.flatten()

        # State rollout (separate into forced response and full state for DPP compliance)
        x_forced = [np.zeros(4)]
        x_expr = [self.x0_param]
        for k in range(self.N):
            x_forced.append(Ad @ x_forced[k] + Bd_flat * self.u[k])
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

        #  soft crosslink constraints (must be non-negative)                                                                       
        s_lead_pointing = cp.Variable((self.N, 2), nonneg=True)                                                                                          
        s_lead_dist = cp.Variable(self.N, nonneg=True)                                                                                                   
                                                                                                                                                      
        s_trail_pointing = cp.Variable((self.N, 2), nonneg=True)                                                                                         
        s_trail_dist = cp.Variable(self.N, nonneg=True)
        
        # Constraints
        constraints = [
            self.u >= -self.max_thrust_param,
            self.u <= self.max_thrust_param
        ]

        phi_max = config.CROSSLINK_GIMBAL_RANGE_RAD
        r_orbit = np.linalg.norm(x0)
        max_along_track = 2 * r_orbit * np.sin(phi_max)
        min_along_track = config.SAFETY_DISTANCE_M

        # Add crosslink constraints to the pre-compiled problem
        for k in range(1, self.N + 1):
            idx = k - 1
            # Leading neighbor constraints (Soft pointing & max distance, hard safety floor)                                                             
            dx_lead = self.leading_pos_param[idx, 0] - x_expr[k][0]                                                                                      
            dy_lead = self.leading_pos_param[idx, 1] - x_expr[k][1]                                                                                      
            constraints.append(dy_lead * np.sin(phi_max) + s_lead_pointing[idx, 0] >= dx_lead * np.cos(phi_max))                                         
            constraints.append(dy_lead * np.sin(phi_max) + s_lead_pointing[idx, 1] >= -dx_lead * np.cos(phi_max))                                        
            constraints.append(dy_lead - s_lead_dist[idx] <= max_along_track)                                                                            
            constraints.append(dy_lead >= min_along_track) # Hard safety limit                                                                           
                                                                                                                                                             
            # Trailing neighbor constraints (Soft pointing & max distance, hard safety floor)                                                            
            dx_trail = self.trailing_pos_param[idx, 0] - x_expr[k][0]                                                                                    
            dy_trail = self.trailing_pos_param[idx, 1] - x_expr[k][1]                                                                                    
            constraints.append((-dy_trail) * np.sin(phi_max) + s_trail_pointing[idx, 0] >= dx_trail * np.cos(phi_max))                                   
            constraints.append((-dy_trail) * np.sin(phi_max) + s_trail_pointing[idx, 1] >= -dx_trail * np.cos(phi_max))                                  
            constraints.append(-dy_trail - s_trail_dist[idx] <= max_along_track)                                                                         
            constraints.append(-dy_trail >= min_along_track) # Hard safety limit

            # Object avoidance constraints (DPP-compliant scalar products)
            constraints.append(
                self.foreign_n_param_x[idx] * x_forced[k][0] +
                self.foreign_n_param_y[idx] * x_forced[k][1] >= self.foreign_dist_param[idx]
            )

        # Add equal spacing goal
        spacing_weight = config.MPC_W_MATRIX
        for k in range(1, self.N + 1):
            idx = k - 1
            lead_spacing = self.leading_pos_param[idx, 1] - x_expr[k][1]
            trail_spacing = x_expr[k][1] - self.trailing_pos_param[idx, 1]
            # minimize square difference for sign
            cost += spacing_weight * cp.square(lead_spacing - trail_spacing)

        # Soft constraint penalties (forces slacks to remain 0 unless physically impossible)                                                             
        pointing_penalty = config.MPC_SOFT_CONSTRAINT_PENALTY                                                                                                                         
        cost += pointing_penalty * (                                                                                                                     
            cp.sum_squares(s_lead_pointing) + cp.sum_squares(s_lead_dist) +                                                                              
            cp.sum_squares(s_trail_pointing) + cp.sum_squares(s_trail_dist)
        )


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
            

    def solve_mpc(self, time: float, pos: np.ndarray, vel: np.ndarray, mass: float, MAX_THRUST_N: float) -> np.ndarray:
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
            self.max_thrust_param.value = MAX_THRUST_N

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

            # Obstacle avoidance
            foreign_n_val = np.zeros((self.N, 2))                                                                                                        
            foreign_dist_val = np.zeros(self.N)
            foreign_dist_val[:] = -1e6 # default far away

            if self.foreignTrajectory is not None:
                safety_dist = self.safety_distance
                x_free = x0.copy()
                for k in range(1, self.N + 1):
                    idx = k - 1
                    next_time = time + k * self.dt
                    x_free = self.Ad @ x_free
                    # Grab nominal position (close enough approximation to maintain convexity)
                    ref_pos_k_eci, ref_vel_k_eci = self.get_nominal_state(next_time)

                    # foreign object position
                    foreign_pos_k_eci, foreign_vel_k_eci = self.foreignTrajectory.get_state_at_time(next_time)

                    # create lvlh frame
                    foreign_pos_k_lvlh, _ = orbit.eci_to_lvlh(ref_pos_k_eci, ref_vel_k_eci, foreign_pos_k_eci, foreign_vel_k_eci)
                    x_r = foreign_pos_k_lvlh[0]
                    y_r = foreign_pos_k_lvlh[1]
                    
                    # Vector from rocket to free response position
                    dx = x_free[0] - x_r
                    dy = x_free[1] - y_r
                    d_free = np.sqrt(dx**2 + dy**2)

                    if d_free > 0:
                        foreign_n_val[idx, 0] = dx / d_free
                        foreign_n_val[idx, 1] = dy / d_free
                        foreign_dist_val[idx] = safety_dist - d_free

            # Update each parameter in the lists to maintain DPP compliance
            for idx in range(self.N):
                self.foreign_n_param_x[idx].value = foreign_n_val[idx, 0]
                self.foreign_n_param_y[idx].value = foreign_n_val[idx, 1]
                self.foreign_dist_param[idx].value = foreign_dist_val[idx]

            
            self.prob.solve(
                solver=cp.OSQP, 
                warm_start=True, 
                verbose=False,
                max_iter=10000,
                eps_abs=1e-3,
                eps_rel=1e-3
            )

            success = self.prob.status in ["optimal", "optimal_inaccurate"]
            if success:                                                                                             
                self.stored_controls = self.u.value                                                                                                                          
            else:                                                                                                                                            
                if self.stored_controls is not None:
                    # Shift previous plan: roll elements left, append 0 at the end
                    self.stored_controls = np.roll(self.stored_controls, -1)
                    self.stored_controls[-1] = 0.0
                else:
                    self.stored_controls = np.zeros(self.N)
                
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
            #     u >= -MAX_THRUST_N,                                                                                                                            
            #     u <= MAX_THRUST_N                                                                                                                              
            # ]
            # for neighborID, neighborTraj in self.neighboringSatTrajectories.items():
            #     if neighborTraj is not None:
            #         neighbor_pos_0_eci, neighbor_vel_0_eci= neighborTraj.get_state_at_time(time)
            #         neighbor_pos_0_lvlh, neighbor_vel_0_lvlh = orbit.eci_to_lvlh(ref_pos_eci, ref_vel_eci, neighbor_pos_0_eci, neighbor_vel_0_eci)
            #         isLeading = neighbor_pos_0_lvlh[1] > 0
            #         phi_max = config.CROSSLINK_GIMBAL_RANGE_RAD                                                                            
            #         r_orbit = np.linalg.norm(ref_pos_eci)                                                                                                  
            #         max_along_track = 2 * r_orbit * np.sin(phi_max)
            #         min_along_track = config.SAFETY_DISTANCE_M
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