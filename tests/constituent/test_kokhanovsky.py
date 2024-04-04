import numpy as np
import sasktran2 as sk


def _test_scenarios():
    # Spherical single scatter
    config = sk.Config()
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

    viewing_geo.add_ray(sk.GroundViewingSolar(0.6, 0.3, 0.5, 200000))

    wavel = np.array([310, 330, 350, 600])

    atmosphere = sk.Atmosphere(geometry, config, wavelengths_nm=wavel)

    sk.climatology.us76.add_us76_standard_atmosphere(atmosphere)

    atmosphere["rayleigh"] = sk.constituent.Rayleigh()

    scen = []

    scen.append(
        {
            "config": config,
            "geometry": geometry,
            "viewing_geo": viewing_geo,
            "atmosphere": atmosphere,
        }
    )

    # DO Preudo-spherical
    config = sk.Config()
    config.multiple_scatter_source = sk.MultipleScatterSource.DiscreteOrdinates
    config.num_streams = 4
    config.delta_m_scaling = False

    geometry = sk.Geometry1D(
        0.6,
        0,
        6327000,
        altitude_grid,
        sk.InterpolationMethod.LinearInterpolation,
        sk.GeometryType.PseudoSpherical,
    )

    scen.append(
        {
            "config": config,
            "geometry": geometry,
            "viewing_geo": viewing_geo,
            "atmosphere": atmosphere,
        }
    )

    # DO spherical
    config = sk.Config()
    config.multiple_scatter_source = sk.MultipleScatterSource.DiscreteOrdinates
    config.num_streams = 4
    config.delta_m_scaling = False

    geometry = sk.Geometry1D(
        0.6,
        0,
        6327000,
        altitude_grid,
        sk.InterpolationMethod.LinearInterpolation,
        sk.GeometryType.Spherical,
    )

    scen.append(
        {
            "config": config,
            "geometry": geometry,
            "viewing_geo": viewing_geo,
            "atmosphere": atmosphere,
        }
    )

    return scen


def test_kokhanovsky_constant():
    """
    Checks that we can create Kokhanovsky BRDF and use it in a radiative transfer calculation without errors
    """
    scens = _test_scenarios()

    for scen in scens:
        engine = sk.Engine(scen["config"], scen["geometry"], scen["viewing_geo"])

        scen["atmosphere"]["surface"] = sk.constituent.SnowKokhanovsky(
            L=3600000, M=5.5e-8
        )

        _ = engine.calculate_radiance(scen["atmosphere"])


def test_kokhanovsky_wf_scalar():
    """
    Checks the weighting functions for a snow surface relative to the parameters L and M
    """
    scens = _test_scenarios()

    for scen in scens:
        engine = sk.Engine(scen["config"], scen["geometry"], scen["viewing_geo"])

        scen["atmosphere"]["surface"] = sk.constituent.SnowKokhanovsky(
            L=3600000, M=5.5e-8
        )

        radiance = sk.test_util.wf.numeric_wf(
            scen["atmosphere"]["surface"].L,
            0.001,
            engine,
            scen["atmosphere"],
            "wf_surface_L",
        )

        sk.test_util.wf.validate_wf(
            radiance["wf_surface_L"],
            radiance["wf_surface_L_numeric"],
            "surface_wavelength",
            decimal=6,
        )

        radiance = sk.test_util.wf.numeric_wf(
            scen["atmosphere"]["surface"].M,
            0.001,
            engine,
            scen["atmosphere"],
            "wf_surface_M",
        )

        sk.test_util.wf.validate_wf(
            radiance["wf_surface_M"],
            radiance["wf_surface_M_numeric"],
            "surface_wavelength",
            decimal=6,
        )
