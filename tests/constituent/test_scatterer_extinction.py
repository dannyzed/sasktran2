from __future__ import annotations

import numpy as np
import pytest
import sasktran2 as sk


def _test_scenarios():
    config = sk.Config()
    config.multiple_scatter_source = sk.MultipleScatterSource.DiscreteOrdinates
    config.num_streams = 4
    config.delta_m_scaling = False

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
    atmosphere.surface.albedo[:] = 0.3

    scen = []

    scen.append(
        {
            "config": config,
            "geometry": geometry,
            "viewing_geo": viewing_geo,
            "atmosphere": atmosphere,
        }
    )

    return scen


@pytest.mark.skip()
def test_scatterer_extinction_altitude_construction():
    """
    Test that the ExtinctionScatterer class can be constructed and basic functionality works
    """
    alts = np.arange(0, 100001, 1000)
    ext_wavel = 525

    ext_profile = sk.climatology.glossac.stratospheric_background(
        5, 20, alts, ext_wavel
    )

    mie = sk.optical.database.OpticalDatabaseGenericScatterer(
        sk.appconfig.database_root().joinpath("cross_sections/mie/sulfate_test.nc")
    )
    radius = np.ones_like(alts) * 105
    const = sk.constituent.ExtinctionScatterer(
        mie, alts, ext_profile, ext_wavel, lognormal_median_radius=radius
    )

    assert len(const.number_density) == len(alts)
    assert len(const.extinction_per_m) == len(alts)


def test_scatterer_extinction_in_engine():
    scens = _test_scenarios()

    for scen in scens:
        engine = sk.Engine(scen["config"], scen["geometry"], scen["viewing_geo"])

        _ = engine.calculate_radiance(scen["atmosphere"])


def test_scatterer_extinction_wf_native_grid():
    """
    Tests the derivatives for a scatterer on the native grid
    """

    scens = _test_scenarios()

    for scen in scens:
        altitude_grid = scen["atmosphere"].model_geometry.altitudes()
        atmosphere = scen["atmosphere"]

        atmosphere["ozone"] = sk.constituent.VMRAltitudeAbsorber(
            sk.optical.O3DBM(), altitude_grid, np.ones_like(altitude_grid) * 1e-6
        )

        atmosphere["strat_aerosol"] = sk.test_util.scenarios.test_aerosol_constituent(
            altitude_grid, True
        )

        engine = sk.Engine(scen["config"], scen["geometry"], scen["viewing_geo"])

        radiance = sk.test_util.wf.numeric_wf(
            atmosphere["strat_aerosol"].extinction_per_m,
            0.0001,
            engine,
            atmosphere,
            "wf_strat_aerosol_extinction",
        )

        sk.test_util.wf.validate_wf(
            radiance["wf_strat_aerosol_extinction"],
            radiance["wf_strat_aerosol_extinction_numeric"],
            wf_dim="strat_aerosol_altitude",
            decimal=5,
        )


def test_scatterer_radius_wf_native_grid():
    """
    Tests the derivatives for a scatterer on the native grid
    """

    scens = _test_scenarios()

    for scen in scens:
        altitude_grid = scen["atmosphere"].model_geometry.altitudes()
        atmosphere = scen["atmosphere"]

        atmosphere["ozone"] = sk.constituent.VMRAltitudeAbsorber(
            sk.optical.O3DBM(), altitude_grid, np.ones_like(altitude_grid) * 1e-6
        )

        atmosphere["strat_aerosol"] = sk.test_util.scenarios.test_aerosol_constituent(
            altitude_grid, True
        )

        engine = sk.Engine(scen["config"], scen["geometry"], scen["viewing_geo"])

        radiance = sk.test_util.wf.numeric_wf(
            atmosphere["strat_aerosol"].lognormal_median_radius,
            0.0001,
            engine,
            atmosphere,
            "wf_strat_aerosol_lognormal_median_radius",
        )

        # Bad precision? Unsure why this is so bad
        sk.test_util.wf.validate_wf(
            radiance["wf_strat_aerosol_lognormal_median_radius"],
            radiance["wf_strat_aerosol_lognormal_median_radius_numeric"],
            wf_dim="strat_aerosol_altitude",
            decimal=2,
        )


def test_scatterer_extinction_wf_interpolated_grid():
    """
    Tests the derivatives for a scatterer on an interpolated grid
    """

    scens = _test_scenarios()

    new_altitudes = np.array([0, 10000, 30000, 70000])

    for scen in scens:
        altitude_grid = scen["atmosphere"].model_geometry.altitudes()
        atmosphere = scen["atmosphere"]

        atmosphere["ozone"] = sk.constituent.VMRAltitudeAbsorber(
            sk.optical.O3DBM(), altitude_grid, np.ones_like(altitude_grid) * 1e-6
        )

        atmosphere["strat_aerosol"] = sk.test_util.scenarios.test_aerosol_constituent(
            new_altitudes, True
        )

        engine = sk.Engine(scen["config"], scen["geometry"], scen["viewing_geo"])

        radiance = sk.test_util.wf.numeric_wf(
            atmosphere["strat_aerosol"].extinction_per_m,
            0.0001,
            engine,
            atmosphere,
            "wf_strat_aerosol_extinction",
        )

        sk.test_util.wf.validate_wf(
            radiance["wf_strat_aerosol_extinction"],
            radiance["wf_strat_aerosol_extinction_numeric"],
            wf_dim="strat_aerosol_altitude",
            decimal=4,
        )


def test_scatterer_numden_wf_interpolated_grid():
    """
    Tests the derivatives for a number density scatterer on an interpolated grid
    """

    scens = _test_scenarios()
    new_altitudes = np.array([0, 10000, 30000, 70000])

    for scen in scens:
        altitude_grid = scen["atmosphere"].model_geometry.altitudes()
        atmosphere = scen["atmosphere"]

        atmosphere["ozone"] = sk.constituent.VMRAltitudeAbsorber(
            sk.optical.O3DBM(), altitude_grid, np.ones_like(altitude_grid) * 1e-6
        )

        atmosphere["strat_aerosol"] = sk.test_util.scenarios.test_aerosol_constituent(
            new_altitudes, False
        )

        engine = sk.Engine(scen["config"], scen["geometry"], scen["viewing_geo"])

        radiance = sk.test_util.wf.numeric_wf(
            atmosphere["strat_aerosol"].number_density,
            0.00001,
            engine,
            atmosphere,
            "wf_strat_aerosol_number_density",
        )

        sk.test_util.wf.validate_wf(
            radiance["wf_strat_aerosol_number_density"],
            radiance["wf_strat_aerosol_number_density_numeric"],
            wf_dim="strat_aerosol_altitude",
            decimal=4,
        )
