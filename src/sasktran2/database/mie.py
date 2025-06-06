from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sasktran2.mie import LinearizedMie
from sasktran2.mie.distribution import (
    ParticleSizeDistribution,
    integrate_mie,
    integrate_mie_cpp,
)
from sasktran2.mie.refractive import RefractiveIndex
from sasktran2.optical.database import (
    OpticalDatabaseGenericScatterer,  # noqa: F401
    OpticalDatabaseGenericScattererRust,
)

from .base import CachedDatabase


class MieDatabase(CachedDatabase, OpticalDatabaseGenericScattererRust):
    def __init__(
        self,
        psize_distribution: ParticleSizeDistribution,
        refractive_index: RefractiveIndex,
        wavelengths_nm: np.array,
        db_root: Path | None = None,
        backend: str = "sasktran2_cpp",
        max_legendre_moments: int = 64,
        num_size_quadrature: int = 1000,
        num_threads=1,
        **kwargs,
    ) -> None:
        """
        A MieDatabase is a database that caches Mie scattering data on the local file system.
        The root directly can optionally be specified, otherwise the default
        folder is used.

        Parameters
        ----------
        psize_distribution : ParticleSizeDistribution
            The particle size distribution
        refractive_index : RefractiveIndex
            The refractive index
        wavelengths_nm : np.array
            The wavelengths to calculate the Mie parameters at
        db_root : Path, optional
            The root directory to store the database, by default None
        backend : str, optional
            The backend to use, by default "sasktran2_cpp". Currently supported
            options are ["sasktran_legacy", "sasktran2", "sasktran2_cpp"]. "sasktran_legacy"
            requires the module `sasktran` to be installed.
        num_threads: int, optional
            The number of threads to use for the calculation, by default 1. Only current works in the sasktran2_cpp backend
        kwargs
            Additional arguments to pass to the particle size distribution, these should match the psize_distribution.args() method
        """

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        hasher = hashlib.sha1()
        encoded = json.dumps(
            {
                **kwargs,
                "wavelengths": wavelengths_nm,
                "max_legendre_moments": max_legendre_moments,
                "backend": backend,
            },
            sort_keys=True,
            cls=NumpyEncoder,
        ).encode()
        hasher.update(encoded)
        identifier = hasher.hexdigest()

        CachedDatabase.__init__(
            self,
            db_root=db_root,
            rel_path=Path("mie")
            .joinpath(refractive_index.identifier)
            .joinpath(psize_distribution.identifier),
        )

        self._data_file = self._db_root.joinpath(f"{identifier}.nc")

        self._psize_args = psize_distribution.args()
        self._psize_dist = psize_distribution
        self._refractive_index = refractive_index
        self._wavelengths_nm = wavelengths_nm
        self._backend = backend
        self._max_legendre_moments = max_legendre_moments
        self._num_threads = num_threads
        self._num_size_quadrature = num_size_quadrature

        for arg in kwargs:
            if arg not in self._psize_args:
                msg = f"Invalid argument {arg} for particle size distribution"
                raise ValueError(msg)

        self._kwargs = kwargs

        OpticalDatabaseGenericScattererRust.__init__(self, self.path())

    def generate(self):
        if self._backend == "sasktran_legacy":
            self._generate_sasktran_legacy()
        elif self._backend == "sasktran2":
            self._generate_sasktran2()
        elif self._backend == "sasktran2_cpp":
            self._generate_sasktran2_cpp()
        elif self._backend == "sasktran2_rust":
            self._generate_sasktran2_rust()
        else:
            msg = f"Invalid backend {self._backend}"
            raise ValueError(msg)

    def _generate_sasktran_legacy(self):
        """
        Generates the data file from legacy sasktran
        """
        try:
            from sasktran import MieWiscombe
            from sasktran.legendre import compute_greek_coefficients_legendre
            from sasktran.mie.distribution import integrated_mie
        except ImportError as err:
            msg = "sasktran_legacy is required to generate Mie databases, try pip install sasktran"
            raise ImportError(msg) from err

        def _create_table(distribution, refractive_index_fn, wavelengths):
            mie = MieWiscombe(max_legendre_moment=self._max_legendre_moments)

            vals = integrated_mie(
                mie,
                distribution,
                refractive_index_fn,
                wavelengths,
                num_quad=self._num_size_quadrature,
                maxintquantile=0.9999,
            )

            # vals xs is in units of nm^2, convert to cm^2 then to m^2
            vals["xs_total"] *= 1e-14 * 1e-4
            vals["xs_scattering"] *= 1e-14 * 1e-4
            vals["xs_absorption"] *= 1e-14 * 1e-4

            lm_a1 = np.zeros_like(vals["lm_p11"].values)
            lm_a2 = np.zeros_like(lm_a1)
            lm_a3 = np.zeros_like(lm_a1)
            lm_a4 = np.zeros_like(lm_a1)
            lm_b1 = np.zeros_like(lm_a1)
            lm_b2 = np.zeros_like(lm_a1)

            for idx in range(len(vals.wavelength.values)):
                selected = vals.isel(wavelength=idx)

                (
                    lm_a1[idx, :],
                    lm_a2[idx, :],
                    lm_a3[idx, :],
                    lm_a4[idx, :],
                    lm_b1[idx, :],
                    lm_b2[idx, :],
                ) = compute_greek_coefficients_legendre(
                    selected["lm_p11"].values,
                    selected["lm_p12"].values,
                    selected["lm_p11"].values,
                    selected["lm_p33"].values,
                    selected["lm_p34"].values,
                    selected["lm_p33"].values,
                )

            vals["lm_a1"] = (["wavelength", "legendre"], lm_a1)
            vals["lm_a2"] = (["wavelength", "legendre"], lm_a2)
            vals["lm_a3"] = (["wavelength", "legendre"], lm_a3)
            vals["lm_a4"] = (["wavelength", "legendre"], lm_a4)
            vals["lm_b1"] = (["wavelength", "legendre"], lm_b1)
            vals["lm_b2"] = (["wavelength", "legendre"], lm_b2)

            ret_ds = xr.Dataset()

            ret_ds["xs_scattering"] = vals["xs_scattering"]
            ret_ds["xs_total"] = vals["xs_total"]
            ret_ds["lm_a1"] = vals["lm_a1"]
            ret_ds["lm_a2"] = vals["lm_a2"]
            ret_ds["lm_a3"] = vals["lm_a3"]
            ret_ds["lm_a4"] = vals["lm_a4"]
            ret_ds["lm_b1"] = vals["lm_b1"]
            ret_ds["lm_b2"] = vals["lm_b2"]

            return ret_ds

        refractive = self._refractive_index.refractive_index_fn

        entries = []
        for vals in product(*self._kwargs.values()):
            psize_args = dict(zip(self._kwargs.keys(), vals, strict=True))

            dist = self._psize_dist.distribution(**psize_args)

            entry = _create_table(dist, refractive, self._wavelengths_nm)

            for key, val in psize_args.items():
                entry[key] = val

            entries.append(entry)

        if len(self._kwargs) > 1:
            # Have to do a multi-index unstack
            ds = (
                xr.concat(entries, dim="args")
                .set_index(args=list(self._kwargs.keys()))
                .unstack("args")
                .rename_dims({"wavelength": "wavelength_nm"})
                .rename_vars({"wavelength": "wavelength_nm"})
            )
        elif len(self._kwargs) == 1:
            ds = (
                xr.concat(entries, dim=next(iter(self._kwargs.keys())))
                .rename_dims({"wavelength": "wavelength_nm"})
                .rename_vars({"wavelength": "wavelength_nm"})
            )
        else:
            # length is 0
            ds = (
                entries[0]
                .rename_dims({"wavelength": "wavelength_nm"})
                .rename_vars({"wavelength": "wavelength_nm"})
            )

        ds.to_netcdf(self._data_file)

    def _generate_sasktran2(self):
        """
        Generates the data file from sasktran2
        """

        def _create_table(distribution, refractive_index_fn, wavelengths):

            mie = LinearizedMie()
            vals = integrate_mie(
                mie,
                distribution,
                refractive_index_fn,
                wavelengths,
                num_quad=self._num_size_quadrature,
                maxintquantile=0.999999,
                compute_coeffs=True,
                num_coeffs=self._max_legendre_moments,
            )
            # vals xs is in units of nm^2, convert to cm^2 then to m^2
            vals["xs_total"] *= 1e-14 * 1e-4
            vals["xs_scattering"] *= 1e-14 * 1e-4
            vals["xs_absorption"] *= 1e-14 * 1e-4

            return vals

        refractive = self._refractive_index.refractive_index_fn

        entries = []
        for vals in product(*self._kwargs.values()):
            psize_args = dict(zip(self._kwargs.keys(), vals, strict=True))

            dist = self._psize_dist.distribution(**psize_args)

            entry = _create_table(dist, refractive, self._wavelengths_nm)

            for key, val in psize_args.items():
                entry[key] = val

            entries.append(entry)

        if len(self._kwargs) > 1:
            # Have to do a multi-index unstack
            ds = (
                xr.concat(entries, dim="args")
                .set_index(args=list(self._kwargs.keys()))
                .unstack("args")
                .rename_dims({"wavelength": "wavelength_nm"})
                .rename_vars({"wavelength": "wavelength_nm"})
            )
        elif len(self._kwargs) == 1:
            ds = (
                xr.concat(entries, dim=next(iter(self._kwargs.keys())))
                .rename_dims({"wavelength": "wavelength_nm"})
                .rename_vars({"wavelength": "wavelength_nm"})
            )
        else:
            # length is 0
            ds = (
                entries[0]
                .rename_dims({"wavelength": "wavelength_nm"})
                .rename_vars({"wavelength": "wavelength_nm"})
            )

        ds.to_netcdf(self._data_file)

    def _generate_sasktran2_cpp(self):
        """
        Generates the data file from sasktran2 using the cpp integration option rather than python
        """
        refractive = self._refractive_index.refractive_index_fn

        prob_dists = []
        for vals in product(*self._kwargs.values()):
            psize_args = dict(zip(self._kwargs.keys(), vals, strict=True))

            prob_dists.append(self._psize_dist.distribution(**psize_args))

        ds = integrate_mie_cpp(
            prob_dists,
            refractive,
            self._wavelengths_nm,
            num_quad=31,
            maxintquantile=0.99999,
            num_coeffs=self._max_legendre_moments,
            num_threads=self._num_threads,
        )

        if len(self._kwargs) > 1:
            multi = pd.MultiIndex.from_product(
                [self._kwargs[k] for k in self._kwargs],
                names=list(self._kwargs.keys()),
            )

            ds = ds.assign_coords(
                xr.Coordinates.from_pandas_multiindex(multi, "distribution")
            )

            # Have to do a multi-index unstack
            ds = ds.unstack("distribution")

        elif len(self._kwargs) == 1:
            ds = ds.rename_dims({"distribution": next(iter(self._kwargs.keys()))})
            ds = ds.drop_vars("distribution")
            for k, v in self._kwargs.items():
                ds.coords[k] = v
        else:
            # length is 0
            ds = ds.isel(distribution=0)

        ds.to_netcdf(self._data_file)

    def clear(self):
        if self._data_file.exists():
            self._data_file.unlink()

    def path(self, key: str | None = None, **kwargs) -> Path | None:  # noqa: ARG002
        if not self._data_file.exists():
            self.generate()
        return self._data_file

    def load_ds(self, key: str | None = None, **kwargs) -> xr.Dataset:
        return xr.open_dataset(self.path(key, **kwargs))
