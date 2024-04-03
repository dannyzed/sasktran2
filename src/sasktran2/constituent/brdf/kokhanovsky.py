from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sasktran2.mie.refractive import RefractiveIndex

from sasktran2 import Atmosphere
from sasktran2.atmosphere import (
    NativeGridDerivative,
    SurfaceDerivativeMapping,
)

from ..base import Constituent
from . import (
    SnowKokhanovskyStokes_1,
    SnowKokhanovskyStokes_3,
    WavelengthInterpolatorMixin,
)


class SnowKokhanovsky(Constituent, WavelengthInterpolatorMixin):
    def __init__(
        self,
        L: np.array,
        M: np.array,
        refractive_index_fn: "RefractiveIndex" = None,
        wavelengths_nm: np.array = None,
        out_of_bounds_mode="zero",
    ) -> None:
        """
        Parameters
        ----------
        wavelengths_nm : np.array, optional
            Wavelengths in [nm] that the albedo is specified at, by default None
        out_of_bounds_mode : str, optional
            One of ["extend" or "zero"], "extend" will extend the last/first value if we are
            interpolating outside the grid. "zero" will set the albedo to 0 outside of the
            grid boundaries, by default "zero"
        """
        Constituent.__init__(self)
        WavelengthInterpolatorMixin.__init__(
            self,
            wavelengths_nm=wavelengths_nm,
            wavenumbers_cminv=None,
            out_of_bounds_mode=out_of_bounds_mode,
            param_length=len(np.atleast_1d(L)),
        )
        self._L = np.atleast_1d(L)
        self._M = np.atleast_1d(M)

        if refractive_index_fn is None:
            from sasktran2.mie.refractive import Ice

            self._refractive_index_fn = Ice()
        else:
            self._refractive_index_fn = refractive_index_fn

    @property
    def L(self) -> np.array:
        return self._L

    @L.setter
    def L(self, L: np.array):
        self._L = np.atleast_1d(L)

    def add_to_atmosphere(self, atmo: Atmosphere):
        if atmo.wavelengths_nm is None:
            msg = (
                "Atmosphere must have wavelengths defined before using SnowKokhonovsky"
            )
            raise ValueError(msg)

        atmo.surface.brdf = (
            SnowKokhanovskyStokes_1()
            if atmo.nstokes == 1
            else SnowKokhanovskyStokes_3()
        )

        interp_matrix = self._interpolating_matrix(atmo)

        # args(0) is (chi + M) * L / wavelength_nm
        # Where chi is the imaginary part of the ice refractive index
        chi = -self._refractive_index_fn.refractive_index(atmo.wavelengths_nm).imag

        M_interp = interp_matrix @ self._M
        L_interp = interp_matrix @ self._L

        atmo.surface.brdf_args[0, :] = (chi + M_interp) * L_interp / atmo.wavelengths_nm
        # atmo.surface.d_brdf_args[0][0, :] = 1

    def register_derivative(self, atmo: Atmosphere, name: str):
        # Start by constructing the interpolation matrix
        interp_matrix = self._interpolating_matrix(atmo)

        derivs = {}

        derivs["albedo"] = SurfaceDerivativeMapping(
            NativeGridDerivative(d_albedo=np.ones(atmo.num_wavel)),
            interpolating_matrix=interp_matrix,
            interp_dim="wavelength",
            result_dim=f"{name}_wavelength",
        )

        return derivs
