from __future__ import annotations

from copy import copy

import numpy as np
import sasktran2 as sk
from sasktran2.optical.database import OpticalDatabaseGenericScatterer


def _storage_delta(atmosphere: sk.Atmosphere, constituent):
    ssa = copy(atmosphere.storage.ssa)
    ext = copy(atmosphere.storage.total_extinction)
    legendre = copy(atmosphere.storage.leg_coeff)

    constituent.add_to_atmosphere(atmosphere)

    return (
        copy(atmosphere.storage.ssa) - ssa,
        copy(atmosphere.storage.total_extinction) - ext,
        copy(atmosphere.storage.leg_coeff) - legendre,
    )


def _test_scenarios():
    config = sk.Config()
    config.multiple_scatter_source = sk.MultipleScatterSource.DiscreteOrdinates
    config.num_streams = 2

    altitude_grid = np.arange(0, 65001, 1000.0)

    geometry = sk.Geometry1D(
        0.6,
        0,
        6327000,
        altitude_grid,
        sk.InterpolationMethod.LinearInterpolation,
        sk.GeometryType.Spherical,
    )

    viewing_geo = sk.ViewingGeometry()

    for tan_alt in np.arange(10000, 60000, 2000):
        viewing_geo.add_ray(sk.TangentAltitudeSolar(tan_alt, 0, 600000, 0.6))

    wavel = np.array([310, 330, 350, 600])

    atmosphere = sk.Atmosphere(geometry, config, wavelengths_nm=wavel)

    sk.climatology.us76.add_us76_standard_atmosphere(atmosphere)

    atmosphere["rayleigh"] = sk.constituent.Rayleigh()

    vmr_altitude_grid = np.array([10000.0, 30000.0, 60000.0])
    atmosphere["ozone"] = sk.constituent.VMRAltitudeAbsorber(
        sk.optical.O3DBM(), vmr_altitude_grid, np.ones_like(vmr_altitude_grid) * 1e-6
    )
    atmosphere.surface.albedo[:] = 0.3

    atmos = []
    atmos.append(atmosphere)

    scen = []

    for atmo in atmos:
        scen.append(
            {
                "config": config,
                "geometry": geometry,
                "viewing_geo": viewing_geo,
                "atmosphere": atmo,
            }
        )

    return scen


def test_scattering_db_construction():
    mie = sk.optical.database.OpticalDatabaseGenericScatterer(
        sk.appconfig.database_root().joinpath("cross_sections/mie/sulfate_test.nc")
    )

    config = sk.Config()

    altitude_grid = np.arange(0, 65001, 1000)

    geometry = sk.Geometry1D(
        0.6,
        0,
        6327000,
        altitude_grid,
        sk.InterpolationMethod.LinearInterpolation,
        sk.GeometryType.Spherical,
    )

    wavel = np.arange(350, 500, 0.1)

    atmosphere = sk.Atmosphere(geometry, config, wavelengths_nm=wavel)

    sk.climatology.us76.add_us76_standard_atmosphere(atmosphere)

    _ = mie.atmosphere_quantities(
        atmo=atmosphere,
        lognormal_median_radius=np.ones_like(atmosphere.model_geometry.altitudes())
        * 105,
    ).extinction[10, :]


def test_scattering_db_wf():
    """
    Tests that weighting functions with respect to the scattering DB auxillary variables (e.g. particle size)
    is working correctly

    The precision of these weighting functions can be quite poor
    """
    scens = _test_scenarios()

    for scen in scens:
        atmosphere = scen["atmosphere"]

        atmosphere["strat_aerosol"] = sk.test_util.scenarios.test_aerosol_constituent(
            atmosphere.model_geometry.altitudes(), extinction_space=False
        )

        engine = sk.Engine(scen["config"], scen["geometry"], scen["viewing_geo"])

        atmosphere["strat_aerosol"].lognormal_median_radius[:] = 107

        radiance = sk.test_util.wf.numeric_wf(
            atmosphere["strat_aerosol"].lognormal_median_radius,
            0.0001,
            engine,
            atmosphere,
            "wf_strat_aerosol_lognormal_median_radius",
        )

        sk.test_util.wf.validate_wf(
            radiance["wf_strat_aerosol_lognormal_median_radius"],
            radiance["wf_strat_aerosol_lognormal_median_radius_numeric"],
            wf_dim="strat_aerosol_altitude",
            decimal=3,
        )


def test_scattering_db_wf_extinction():
    """
    Tests that weighting functions with respect to the scattering DB auxillary variables (e.g. particle size)
    is working correctly

    The precision of these weighting functions can be quite poor
    """
    scens = _test_scenarios()

    for scen in scens:
        atmosphere = scen["atmosphere"]

        atmosphere["strat_aerosol"] = sk.test_util.scenarios.test_aerosol_constituent(
            atmosphere.model_geometry.altitudes(), extinction_space=True
        )

        engine = sk.Engine(scen["config"], scen["geometry"], scen["viewing_geo"])

        atmosphere["strat_aerosol"].lognormal_median_radius[:] = 107

        radiance = sk.test_util.wf.numeric_wf(
            atmosphere["strat_aerosol"].lognormal_median_radius,
            0.000001,
            engine,
            atmosphere,
            "wf_strat_aerosol_lognormal_median_radius",
        )

        sk.test_util.wf.validate_wf(
            radiance["wf_strat_aerosol_lognormal_median_radius"],
            radiance["wf_strat_aerosol_lognormal_median_radius_numeric"],
            wf_dim="strat_aerosol_altitude",
            decimal=2,
        )


def test_scattering_db_rust_impl():
    """
    Tests that the scattering db rust implementation agrees with the Python reference implementation
    """
    scens = _test_scenarios()

    alt_grid = np.arange(0, 65001, 1000.0)
    ext = np.zeros(len(alt_grid))
    ext[:] = 1e-7

    wavel = np.array([310, 330, 350, 600, 745.0])

    r_constituents = []
    py_constituents = []

    db2 = sk.database.MieDatabase(
        sk.mie.distribution.LogNormalDistribution(),
        sk.mie.refractive.H2SO4(),
        wavel,
        median_radius=np.array([100, 200]),
        mode_width=np.array([1.5, 1.6, 1.7]),
    )

    r_constituents.append(
        sk.constituent.ExtinctionScatterer(
            db2,
            alt_grid,
            ext,
            745.0,
            median_radius=np.ones_like(ext) * 120.0,
            mode_width=np.ones_like(ext) * 1.55,
        )
    )

    py_constituents.append(
        sk.constituent.ExtinctionScatterer(
            OpticalDatabaseGenericScatterer(db2.path()),
            alt_grid,
            ext,
            745.0,
            median_radius=np.ones_like(ext) * 120.0,
            mode_width=np.ones_like(ext) * 1.55,
        )
    )

    db1 = sk.database.MieDatabase(
        sk.mie.distribution.LogNormalDistribution().freeze(mode_width=1.6),
        sk.mie.refractive.H2SO4(),
        wavel,
        median_radius=np.array([100, 200]),
    )

    r_constituents.append(
        sk.constituent.ExtinctionScatterer(
            db1,
            alt_grid,
            ext,
            745.0,
            median_radius=np.ones_like(ext) * 120.0,
        )
    )

    py_constituents.append(
        sk.constituent.ExtinctionScatterer(
            OpticalDatabaseGenericScatterer(db1.path()),
            alt_grid,
            ext,
            745.0,
            median_radius=np.ones_like(ext) * 120.0,
        )
    )

    db0 = sk.database.MieDatabase(
        sk.mie.distribution.LogNormalDistribution().freeze(
            mode_width=1.6, median_radius=120
        ),
        sk.mie.refractive.H2SO4(),
        wavel,
    )

    r_constituents.append(sk.constituent.ExtinctionScatterer(db0, alt_grid, ext, 745.0))

    py_constituents.append(
        sk.constituent.ExtinctionScatterer(
            OpticalDatabaseGenericScatterer(db0.path()), alt_grid, ext, 745.0
        )
    )

    for scen in scens:
        atmosphere = scen["atmosphere"]

        for r_const, py_const in zip(r_constituents, py_constituents, strict=False):

            r_ssa, r_k, r_leg = _storage_delta(atmosphere, r_const)
            py_ssa, py_k, py_leg = _storage_delta(atmosphere, py_const)

            np.testing.assert_allclose(
                r_ssa,
                py_ssa,
                rtol=1e-5,
                atol=1e-5,
                err_msg="SSA does not match between Rust and Python",
            )

            np.testing.assert_allclose(
                r_k,
                py_k,
                rtol=1e-5,
                atol=1e-5,
                err_msg="Extinction does not match between Rust and Python",
            )

            np.testing.assert_allclose(
                r_leg,
                py_leg,
                rtol=1e-5,
                atol=1e-5,
                err_msg="Legendre coefficients do not match between Rust and Python",
            )
