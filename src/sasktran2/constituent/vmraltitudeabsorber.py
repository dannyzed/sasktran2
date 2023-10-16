import numpy as np

from sasktran2 import Atmosphere
from sasktran2.atmosphere import (
    InterpolatedDerivativeMapping,
    NativeGridDerivative,
)
from sasktran2.optical import pressure_temperature_to_numberdensity
from sasktran2.optical.base import OpticalProperty
from sasktran2.util.interpolation import linear_interpolating_matrix

from .base import Constituent


class VMRAltitudeAbsorber(Constituent):
    def __init__(
        self,
        optical_property: OpticalProperty,
        altitudes_m: np.array,
        vmr: np.array,
        out_of_bounds_mode: str = "zero",
    ) -> None:
        """
        An atmospheric constituent that is specified through volume mixing ratio (VMR) on an altitude grid.
        The altitude grid need not match the global atmospheric grid, in the case they are different the VMRs
        will be linearly interpolated to the atmospheric grid locations.

        The constituent has an associated OpticalProperty that defines the absorption cross section as a function
        of wavelength.

        Parameters
        ----------
        optical_property : OpticalProperty
            An object that provides absorption cross sections as a function of wavelength
        altitudes_m : np.array
            Altitudes in [m] that the VMR is specified on
        vmr : np.array
            Volume mixing ratio
        out_of_bounds_mode : str, optional
            One of ['zero', 'extend'], 'zero' will set the VMR to be 0 on any altitudes that are
            out of bounds.  'extend' will extend the last or first VMR value.  By default 'zero'
        """
        super().__init__()

        self._out_of_bounds_mode = out_of_bounds_mode
        self._altitudes_m = altitudes_m
        self._vmr = vmr
        self._optical_property = optical_property

    @property
    def vmr(self):
        return self._vmr

    @vmr.setter
    def vmr(self, vmr: np.array):
        self._vmr = vmr

    def add_to_atmosphere(self, atmo: Atmosphere):
        self._optical_quants = self._optical_property.atmosphere_quantities(atmo)

        if atmo.pressure_pa is None or atmo.temperature_k is None:
            msg = "Both pressure_pa and temperature_k have to be specified in the atmosphere to use VMRAltitudeAbsorber"
            raise ValueError(msg)

        number_density = pressure_temperature_to_numberdensity(
            atmo.pressure_pa, atmo.temperature_k
        )

        interp_matrix = linear_interpolating_matrix(
            self._altitudes_m,
            atmo.model_geometry.altitudes(),
            self._out_of_bounds_mode.lower(),
        )

        interp_vmr = interp_matrix @ self._vmr

        atmo.storage.total_extinction += (
            self._optical_quants.extinction
            * (number_density * interp_vmr)[:, np.newaxis]
        )

    def register_derivative(self, atmo: Atmosphere, name: str):
        number_density = pressure_temperature_to_numberdensity(
            atmo.pressure_pa, atmo.temperature_k
        )
        interp_matrix = linear_interpolating_matrix(
            self._altitudes_m,
            atmo.model_geometry.altitudes(),
            self._out_of_bounds_mode.lower(),
        )
        derivs = {}

        derivs["vmr"] = InterpolatedDerivativeMapping(
            NativeGridDerivative(
                d_extinction=self._optical_quants.extinction
                * number_density[:, np.newaxis],
                d_ssa=self._optical_quants.extinction
                * (self._optical_quants.ssa - atmo.storage.ssa)
                / atmo.storage.total_extinction
                * number_density[:, np.newaxis],
            ),
            interpolating_matrix=interp_matrix,
            interp_dim="altitude",
            result_dim=f"{name}_altitude",
        )

        return derivs
