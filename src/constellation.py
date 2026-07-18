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
        print("Compiling satellite controllers...", flush=True)
        for i in range(number_of_satellites):
            print(f"  -> Compiling Controller for Satellite {i}...", flush=True)
            degrees_long = (360/number_of_satellites) * i  # Evenly spaced longitudes
            sat = Satellite(id = i, mass = 500, altitude = altitude, longitude = degrees_long)

            satellite.append(sat)

        # Establish order and crosslinks
        from src.vehicle.crosslink import Crosslink
        for i, sat in enumerate(satellite):
            sat.leadingSat = satellite[(i + 1) % number_of_satellites]                                                                                                    
            sat.trailingSat = satellite[(i - 1) % number_of_satellites] 
            
            # Create physical crosslink objects
            sat.leading_link = Crosslink(sender=sat, receiver=sat.leadingSat)
            sat.trailing_link = Crosslink(sender=sat, receiver=sat.trailingSat)

        # Establish initial connection status
        for sat in satellite:
            sat.leadingConnection = sat.leading_link.check_connection()
            sat.trailingConnection = sat.trailing_link.check_connection()

        return satellite
    
    def check_crosslinks(self):
        """
        Checks the crosslink connections for all satellites in the constellation.
        """
        for sat in self.satellites:
            sat.leadingConnection = sat.leading_link.check_connection()
            sat.trailingConnection = sat.trailing_link.check_connection()

    def communicate(self):
        """
        Transmits data between sats if crosslink is active
        """
        for sat in self.satellites:
            sat.leading_link.transmit()
            sat.trailing_link.transmit()