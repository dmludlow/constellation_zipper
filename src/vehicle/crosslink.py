from typing import TYPE_CHECKING
import numpy as np
import src.simulation.config as config

if TYPE_CHECKING:
    from src.vehicle.satellite import Satellite
    from src.physics.trajectory import Trajectory

class Crosslink:
    """
    Represents the physical and communication connection between two satellites.
    """

    sender: "Satellite"
    receiver: "Satellite"
    gimbal_range: float = config.CROSSLINK_GIMBAL_RANGE_RAD
    connection_active: bool = False

    def __init__(self, sender: "Satellite", receiver: "Satellite"):
        self.sender = sender
        self.receiver = receiver

    def check_connection(self) -> bool:
        """
        Calculates the pointing vector geometry. Updates and returns connection_active.
        """
        pos_send, vel_send = self.sender.positionECI, self.sender.velocityECI
        pos_recv = self.receiver.positionECI

        # Relative direction vector
        relative_pos = pos_recv - pos_send
        distance = np.linalg.norm(relative_pos)
        if distance == 0:
            self.connection_active = False
            return False

        relative_unit = relative_pos / distance
        pointing_vector = vel_send / np.linalg.norm(vel_send)

        # Calculate viewing angle (leading vs trailing chord alignment)
        dot_product = relative_unit.dot(pointing_vector)
        if dot_product > 0:  # Leading
            viewing_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        else:  # Trailing
            viewing_angle = np.arccos(np.clip(-dot_product, -1.0, 1.0))

        # Update link status
        self.connection_active = (viewing_angle <= self.gimbal_range)
        return self.connection_active

    def transmit(self) -> bool:
        """
        Transmits the sender's predicted trajectory to the receiver if the link is active.
        """
        if self.check_connection():
            if self.sender.controller.computedTrajectory is not None:
                self.receiver.controller.receive_neighboring_trajectories(
                    self.sender.id, 
                    self.sender.controller.computedTrajectory
                )
                return True
        return False
