import numpy as np
import xarray as xr

import sasktran2 as sk


def validate_wf(analytic, numerical, wf_dim="altitude", decimal=6):
    max_by_alt = np.abs(analytic).max(dim=wf_dim)

    max_by_alt.to_numpy()[max_by_alt.to_numpy() == 0] = 1e99

    rel_diff = (analytic - numerical) / max_by_alt

    np.testing.assert_array_almost_equal(rel_diff, 0, decimal=decimal)


def numeric_wf(
    input_var: np.array,
    fractional_change: float,
    engine: sk.Engine,
    atmosphere: sk.Atmosphere,
    analytic_wf_name: str,
) -> xr.Dataset:
    base_radiance = engine.calculate_radiance(atmosphere)

    central_diff_wf = np.zeros_like(base_radiance[analytic_wf_name].to_numpy())

    for i in range(len(input_var)):
        dx = input_var[i] * fractional_change

        input_var[i] += dx

        radiance_above = engine.calculate_radiance(atmosphere)

        input_var[i] -= 2 * dx

        radiance_below = engine.calculate_radiance(atmosphere)

        input_var[i] += dx

        central_diff_wf[i] = (
            radiance_above["radiance"] - radiance_below["radiance"]
        ) / (2 * dx)

    base_radiance[analytic_wf_name + "_numeric"] = (
        base_radiance[analytic_wf_name].dims,
        central_diff_wf,
    )

    return base_radiance
