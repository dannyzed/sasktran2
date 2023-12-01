import numpy as np
import xarray as xr

import sasktran2 as sk


class StateCombiner:
    def __init__(
        self, k: np.array, ssa: np.array, leg_coeff: np.array, albedo: np.array
    ) -> None:
        pass


class PCAEngine(sk.Engine):
    def __init__(
        self,
        config: sk.Config,
        model_geometry: sk.Geometry1D,
        viewing_geo: sk.ViewingGeometry,
        min_pca_variance=0.995,
        max_pca_components=20,
        wavelength_nm_bins=10,
    ) -> None:
        self._raw_atmo = sk.Atmosphere(model_geometry, config, numwavel=1)
        super().__init__(config, model_geometry, viewing_geo)

        self._min_pca_variance = min_pca_variance
        self._wavelength_nm_bins = wavelength_nm_bins
        self._max_pca_components = max_pca_components

    def calculate_radiance(
        self, atmosphere: sk.Atmosphere, output: sk.Output = None  # noqa: ARG002
    ):
        try:
            from sklearn.decomposition import PCA

            pca = PCA(self._max_pca_components)
        except ImportError:
            msg = "scikit-learn must be installed to use PCAEngine"
            raise OSError(msg)  # noqa: B904

        internal_atmo = atmosphere.internal_object()
        full_X = np.vstack(
            [
                internal_atmo.storage.total_extinction,
                internal_atmo.storage.ssa,
                internal_atmo.storage.leg_coeff[1:].reshape(
                    -1, internal_atmo.storage.leg_coeff.shape[-1]
                ),
            ]
        ).T
        ngeo = internal_atmo.storage.total_extinction.shape[0]

        wavelength_bin_edges = np.unique(
            np.concatenate(
                (
                    np.arange(
                        np.nanmin(atmosphere.wavelengths_nm),
                        np.nanmax(atmosphere.wavelengths_nm),
                        self._wavelength_nm_bins,
                    ),
                    [np.nanmax(atmosphere.wavelengths_nm)],
                )
            )
        )

        wavelength_bins = list(zip(wavelength_bin_edges[:-1], wavelength_bin_edges[1:]))

        rads = []
        for bin in wavelength_bins:
            inside = (atmosphere.wavelengths_nm >= bin[0]) & (
                atmosphere.wavelengths_nm < bin[1]
            )

            pca.fit(full_X[inside, :])

            ## Calculate mean radiance
            mean_X = full_X[inside, :].mean(axis=0)

            self._raw_atmo.storage.total_extinction[:, 0] = mean_X[:ngeo]
            self._raw_atmo.storage.ssa[:, 0] = mean_X[ngeo : 2 * ngeo]
            self._raw_atmo.surface.albedo[0] = np.nanmean(internal_atmo.surface.albedo)
            self._raw_atmo.storage.leg_coeff[1:, :, 0] = mean_X[2 * ngeo :].reshape(
                -1, ngeo
            )
            self._raw_atmo.storage.leg_coeff[0, :, 0] = 1

            raw_rad = super().calculate_radiance(self._raw_atmo).isel(wavelength=0)
            # Get dI/dX

            c_vars = ["wf_extinction", "wf_ssa"]
            for i in range(1, internal_atmo.storage.leg_coeff.shape[0]):
                c_vars.append(f"wf_leg_coeff_{i}")
            K = xr.concat([raw_rad[c] for c in c_vars], dim="altitude")

            dI_pca = K @ xr.DataArray(pca.components_, dims=["pca", "altitude"])
            pca_weights = (
                dI_pca.to_numpy().reshape(-1, self._max_pca_components).mean(axis=0)
            )

            pca_dim = pca.transform(full_X[inside, :])
            pca_distance = np.linalg.norm(pca_dim * pca_weights[np.newaxis, :], axis=1)

            # dX = full_X[inside] - mean_X
            # adjusted_rad = (K @ xr.DataArray(dX, dims=["wavelength", "altitude"]) + raw_rad["radiance"]).assign_coords({"wavelength": atmosphere.wavelengths_nm[inside]})

            adjusted_rad = (
                (
                    dI_pca @ xr.DataArray(pca_dim, dims=["wavelength", "pca"])
                    + raw_rad["radiance"]
                )
                .assign_coords({"wavelength": atmosphere.wavelengths_nm[inside]})
                .to_dataset(name="radiance")
            )
            adjusted_rad["pca_distance"] = (["wavelength"], pca_distance)

            rads.append(adjusted_rad)

        return xr.concat(rads, dim="wavelength")
