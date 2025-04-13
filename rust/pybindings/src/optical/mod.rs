use std::{path::PathBuf, str::FromStr};

use numpy::ndarray::*;
use numpy::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use sk_core::interpolation::linear::Grid;
use sk_core::optical::xsec_dbase::XsecDatabaseInterp;
use sk_core::optical::xsec_dbase::{SKXsecDatabase, read_fwf_folder};

#[pyclass]
/// An absorber database that depends on wavenumber and 1 parameter (e.g. temperature)
///
/// Parameters
/// ----------
/// wavenumber_cminv : ndarray
///   Wavenumber in cm^-1
/// params : ndarray
///   Parameters that the cross section depends on
/// xsec_m2 : ndarray
///   Cross section in m^2/molecule [num_param, num_wavenumber]
pub struct AbsorberDatabaseDim2 {
    db: SKXsecDatabase<Ix2>,
}

#[pyclass]
/// An absorber database that depends on wavenumber and 2 parameters (e.g. temperature and pressure)
///
/// Parameters
/// ----------
/// wavenumber_cminv : ndarray
///  Wavenumber in cm^-1
pub struct AbsorberDatabaseDim3 {
    db: SKXsecDatabase<Ix3>,
}

#[pymethods]
impl AbsorberDatabaseDim2 {
    #[new]
    fn new(
        wavenumber_cminv: PyReadonlyArray1<f64>,
        params: PyReadonlyArray1<f64>,
        xsec_m2: PyReadonlyArray2<f64>,
    ) -> PyResult<Self> {
        let arr_wvnum = wavenumber_cminv.as_array().to_owned();
        let arr_params = params.as_array().to_owned();

        let arr_xsec = xsec_m2.as_array().to_owned();

        let db = SKXsecDatabase::<Ix2>::new(arr_xsec, Grid::new(arr_wvnum), vec![arr_params])
            .ok_or_else(|| PyValueError::new_err("Failed to create SKXsecDatabase"))?;

        Ok(Self { db })
    }

    #[classmethod]
    fn from_fwf_folder(_: &Bound<'_, PyType>, path: String) -> PyResult<Self> {
        let path = PathBuf::from_str(&path)?;

        let db: SKXsecDatabase<Ix2> = read_fwf_folder(path).unwrap().into();
        Ok(Self { db })
    }

    fn cross_section<'py>(
        &self,
        wavenumber_cminv: PyReadonlyArray1<'py, f64>,
        params: PyReadonlyArray1<'py, f64>,
    ) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<f64>>) {
        let arr_wvnum = wavenumber_cminv.as_array();
        let arr_params = params.as_array();

        let arr_xsec = PyArray2::zeros(
            wavenumber_cminv.py(),
            (arr_params.len(), arr_wvnum.len()),
            false,
        );
        let arr_d_xsec = PyArray3::zeros(
            wavenumber_cminv.py(),
            (1, arr_params.len(), arr_wvnum.len()),
            false,
        );

        unsafe {
            let mut mut_view = arr_xsec.as_array_mut();
            let mut mut_d_view = arr_d_xsec.as_array_mut();
            Zip::from(mut_view.rows_mut())
                .and(mut_d_view.axis_iter_mut(Axis(1)))
                .and(arr_params)
                .par_for_each(|mut row, d_row, param| {
                    let params = Array1::from(vec![*param]);

                    let _ = self
                        .db
                        .xs_emplace(&arr_wvnum, &params, &mut row, Some(d_row));
                });
        }

        (arr_xsec, arr_d_xsec)
    }
}

#[pymethods]
impl AbsorberDatabaseDim3 {
    #[new]
    fn new(
        wavenumber_cminv: PyReadonlyArray1<f64>,
        param_0: PyReadonlyArray1<f64>,
        param_1: PyReadonlyArray1<f64>,
        xsec_m2: PyReadonlyArray3<f64>,
    ) -> PyResult<Self> {
        let arr_wvnum = wavenumber_cminv.as_array().to_owned();

        let arr_param_0 = param_0.as_array().to_owned();
        let arr_param_1 = param_1.as_array().to_owned();

        let arr_xsec = xsec_m2.as_array().to_owned();

        let db = SKXsecDatabase::<Ix3>::new(
            arr_xsec,
            Grid::new(arr_wvnum),
            vec![arr_param_0, arr_param_1],
        )
        .ok_or_else(|| PyValueError::new_err("Failed to create SKXsecDatabase"))?;

        Ok(Self { db })
    }

    fn cross_section<'py>(
        &self,
        wavenumber_cminv: PyReadonlyArray1<'py, f64>,
        params: PyReadonlyArray2<'py, f64>,
    ) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<f64>>) {
        let arr_wvnum = wavenumber_cminv.as_array();
        let arr_params = params.as_array();

        let ngeo = arr_params.shape()[1];

        let arr_xsec = PyArray2::zeros(wavenumber_cminv.py(), (ngeo, arr_wvnum.len()), false);
        let arr_d_xsec = PyArray3::zeros(wavenumber_cminv.py(), (2, ngeo, arr_wvnum.len()), false);

        unsafe {
            let mut mut_view = arr_xsec.as_array_mut();
            let mut mut_d_view = arr_d_xsec.as_array_mut();
            Zip::from(mut_view.rows_mut())
                .and(mut_d_view.axis_iter_mut(Axis(1)))
                .and(arr_params.axis_iter(Axis(1)))
                .par_for_each(|mut row, d_row, param| {
                    let _ = self
                        .db
                        .xs_emplace(&arr_wvnum, &param, &mut row, Some(d_row));
                });
        }

        (arr_xsec, arr_d_xsec)
    }
}
