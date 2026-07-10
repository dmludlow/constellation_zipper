from src.vehicle.satellite import Satellite

class Constellation:
    """
    A class representing a constellation of satellites
    """

    satellites: list[Satellite] # List of all satellites in the constellation

    def __init__(self, number_of_satellites: int, altitude: float):
        """
        Initializes a Constellation of specified number of satellites in a circular equitorial orbit at a given altitude.
        """

        self.satellites = self.generate_satellites(number_of_satellites, altitude)


    #TODO
    def generate_satellites(self, number_of_satellites, altitude) -> list[Satellite]:
        """
        Generates a list of satellites in a circular equitorial orbit at a given altitude with equal spacing.

        Currently assumes standard mass of 500 kg)
        """
        satellite = []
        for i in range(number_of_satellites):
            degrees_long = (360/number_of_satellites) * i  # Evenly spaced longitudes
            sat = Satellite(id = i, mass = 500, altitude = altitude, longitude = degrees_long)

            satellite.append(sat)
        return satellite