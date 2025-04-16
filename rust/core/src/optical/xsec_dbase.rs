use crate::interpolation::OutOfBoundsMode;
use crate::interpolation::linear::Grid;
use crate::interpolation::linear::Interp1Weights;
use ndarray::{Data, DataMut};
use ndarray::{Zip, prelude::*};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::ops::AddAssign;
use std::path::PathBuf;

use super::OpticalProperty;

/// Returns back the indices required to sort a vector of f64 values.
pub fn argsort_f64(data: &[f64]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|&i, &j| {
        data[i]
            .partial_cmp(&data[j])
            .unwrap_or(std::cmp::Ordering::Equal) // Handle NaN by treating as equal
    });
    indices
}

/// Returns a sorted array of unique values from the input array.
fn unique_values<S, D>(array: &ArrayBase<S, D>) -> Array1<f64>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    let mut seen = Vec::new();

    for &x in array.iter() {
        if x.is_nan() {
            continue;
        }
        if !seen.contains(&x) {
            seen.push(x);
        }
    }

    seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Array1::from(seen)
}

/// Trait for interpolating cross sections from a database.  This is used to interpolate
/// cross sections from a database to a given wavenumber and set of parameters.
pub trait XsecDatabaseInterp {
    fn xs_emplace<S1, S2, S3>(
        &self,
        wvnum: &ArrayBase<S1, Ix1>,
        params: &ArrayBase<S2, Ix1>,
        xs: &mut ArrayBase<S3, Ix1>,
        d_xs: Option<ArrayViewMut2<'_, f64>>,
    ) -> Result<(), String>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
        S3: DataMut<Elem = f64>;
}

/// Raw storage structure that holds the cross section data from something like the
/// HITRAN fwf files, or the LBLRTM files.  This is basically lists of cross section data as a function
/// of wavenumber at a given temperature and pressure.  We assume wavenumber is specified in
/// vacuum cm^-1, and the cross section is in cm^2/molecule.
pub struct XsecDatabase {
    xsec: Vec<Array1<f64>>,
    wvnum: Vec<Array1<f64>>,
    params: HashMap<String, Vec<f64>>,
}

/// Let's us concatenate two XsecDatabase objects together.  This is useful for combining
/// databases from different pressure/temperatures together
impl AddAssign for XsecDatabase {
    fn add_assign(&mut self, other: Self) {
        self.xsec.extend(other.xsec);
        self.wvnum.extend(other.wvnum);

        for (param, vals) in other.params {
            self.params
                .entry(param)
                .or_insert_with(Vec::new)
                .extend(vals);
        }
    }
}

/// A cross sectional database in the format that SASKTRAN requires it in.  This is
/// a cross sectional array of shape (len(param1), len(param2), ..., len(paramN), len(wvnum))
/// Where parameters are things like pressure, temperature, or foregin broadenerers, wvnum is always
/// The last dimension
pub struct SKXsecDatabase<D: Dimension> {
    xsec: Array<f64, D>,
    wvnum: Grid,
    params: Vec<Array1<f64>>,
    param_names: Vec<String>,
}

impl<D: Dimension> SKXsecDatabase<D> {
    pub fn new(
        xsec: Array<f64, D>,
        wvnum: Grid,
        params: Vec<Array1<f64>>,
        param_names: Vec<String>,
    ) -> Option<Self> {
        assert_eq!(params.len(), D::NDIM? - 1);

        Some(SKXsecDatabase {
            xsec,
            wvnum,
            params,
            param_names,
        })
    }
}

impl From<XsecDatabase> for SKXsecDatabase<Ix2> {
    fn from(db: XsecDatabase) -> Self {
        // We need to figure out how many dimensions we have
        let mut params: Vec<Array1<f64>> = vec![];
        let mut sidx: Vec<Vec<usize>> = vec![];
        for (_, vals) in &db.params {
            let unique_vals = unique_values(&Array1::from_vec(vals.clone()));

            if unique_vals.len() > 1 {
                params.push(Array1::from(unique_vals));
                sidx.push(argsort_f64(&vals));
            }
        }

        // Next we need to create a common wavenumber grid, concatenate all of the wavenumbers together
        // and then sort them
        let mut all_wvnum: Vec<f64> = vec![];
        for wvnum in &db.wvnum {
            all_wvnum.extend(wvnum.iter());
        }
        let unique_wvnum = unique_values(&Array1::from(all_wvnum));

        let mut xsec = Array2::zeros((params[0].len(), unique_wvnum.len()));

        for i in 0..params[0].len() {
            xsec.slice_mut(s![i, ..]).assign(&db.xsec[sidx[0][i]]);
        }

        SKXsecDatabase::<Ix2>::new(
            xsec,
            Grid::new(unique_wvnum),
            params,
            vec!["temperature_k".to_string()],
        )
        .unwrap()
    }
}

impl XsecDatabaseInterp for SKXsecDatabase<Ix2> {
    fn xs_emplace<S1, S2, S3>(
        &self,
        wvnum: &ArrayBase<S1, Ix1>,
        params: &ArrayBase<S2, Ix1>,
        xs: &mut ArrayBase<S3, Ix1>,
        d_xs: Option<ArrayViewMut2<'_, f64>>,
    ) -> Result<(), String>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
        S3: DataMut<Elem = f64>,
    {
        let params_slice = &self.params[0];

        let weights = params_slice.interp1_weights(params[0], OutOfBoundsMode::Extend);

        for (i, weight, _) in weights.iter() {
            let internal_xs = self.xsec.slice(s![*i, ..]);

            Zip::indexed(wvnum).for_each(|j, wv| {
                let wvnum_weights = &self.wvnum.interp1_weights(*wv, OutOfBoundsMode::Zero);
                let local_xs = internal_xs[wvnum_weights[0].0] * wvnum_weights[0].1
                    + internal_xs[wvnum_weights[1].0] * wvnum_weights[1].1;

                xs[j] += local_xs * *weight;
            });
        }

        if let Some(mut d_xs_mut) = d_xs {
            for (i, _, d_weight) in weights.iter() {
                let internal_xs = self.xsec.slice(s![*i, ..]);
                Zip::indexed(wvnum).for_each(|j, wv| {
                    let wvnum_weights = &self.wvnum.interp1_weights(*wv, OutOfBoundsMode::Zero);
                    let local_xs = internal_xs[wvnum_weights[0].0] * wvnum_weights[0].1
                        + internal_xs[wvnum_weights[1].0] * wvnum_weights[1].1;

                    d_xs_mut[[0, j]] += local_xs * *d_weight;
                });
            }
        }

        // Have to iterate through all of the
        Ok(())
    }
}

impl XsecDatabaseInterp for SKXsecDatabase<Ix3> {
    fn xs_emplace<S1, S2, S3>(
        &self,
        wvnum: &ArrayBase<S1, Ix1>,
        params: &ArrayBase<S2, Ix1>,
        xs: &mut ArrayBase<S3, Ix1>,
        d_xs: Option<ArrayViewMut2<'_, f64>>,
    ) -> Result<(), String>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
        S3: DataMut<Elem = f64>,
    {
        let weights_0 = &self.params[0].interp1_weights(params[0], OutOfBoundsMode::Extend);
        let weights_1 = &self.params[1].interp1_weights(params[1], OutOfBoundsMode::Extend);

        for (i, weight_0, _) in weights_0.iter() {
            for (j, weight_1, _) in weights_1.iter() {
                let internal_xs = self.xsec.slice(s![*i, *j, ..]);
                Zip::indexed(wvnum).for_each(|k, wv| {
                    let wvnum_weights = &self.wvnum.interp1_weights(*wv, OutOfBoundsMode::Zero);
                    let local_xs = internal_xs[wvnum_weights[0].0] * wvnum_weights[0].1
                        + internal_xs[wvnum_weights[1].0] * wvnum_weights[1].1;

                    xs[k] += local_xs * *weight_0 * *weight_1;
                });
            }
        }

        if let Some(mut d_xs_mut) = d_xs {
            for (i, weight_0, d_weight_0) in weights_0.iter() {
                for (j, weight_1, d_weight_1) in weights_1.iter() {
                    let internal_xs = self.xsec.slice(s![*i, *j, ..]);
                    Zip::indexed(wvnum).for_each(|k, wv| {
                        let wvnum_weights = &self.wvnum.interp1_weights(*wv, OutOfBoundsMode::Zero);
                        let xs_interp = internal_xs[wvnum_weights[0].0] * wvnum_weights[0].1
                            + internal_xs[wvnum_weights[1].0] * wvnum_weights[1].1;

                        d_xs_mut[[0, k]] += xs_interp * *d_weight_0 * *weight_1;
                        d_xs_mut[[1, k]] += xs_interp * *weight_0 * *d_weight_1;
                    });
                }
            }
        }

        // Have to iterate through all of the
        Ok(())
    }
}

impl OpticalProperty for SKXsecDatabase<Ix2> {
    fn optical_quantities_emplace(
        &self,
        inputs: &dyn crate::constituent::StorageInputs,
        optical_quantities: &mut super::OpticalQuantities,
    ) -> anyhow::Result<()> {
        let wavenumber_cminv = inputs
            .get_parameter("wavenumbers_cminv")
            .ok_or_else(|| anyhow::anyhow!("Wavenumber not found in inputs"))?;

        let param = inputs
            .get_parameter(&self.param_names[0])
            .ok_or_else(|| anyhow::anyhow!("Parameter not found in inputs"))?;

        let _ = optical_quantities.resize(param.len(), wavenumber_cminv.len());

        let xs = &mut optical_quantities.cross_section;

        Zip::from(xs.rows_mut())
            .and(param)
            .par_for_each(|mut row, param| {
                let params = Array1::from(vec![*param]);

                let _ = self.xs_emplace(&wavenumber_cminv, &params, &mut row, None);
            });

        Ok(())
    }

    fn optical_derivatives_emplace(
        &self,
        inputs: &dyn crate::constituent::StorageInputs,
        d_optical_quantities: &mut HashMap<String, super::OpticalQuantities>,
    ) -> anyhow::Result<()> {
        todo!()
    }
}

#[allow(dead_code)]
struct Header {
    pub short_mol_name: String,
    pub wvnum_start: f64,
    pub wvnum_end: f64,
    pub num_points: i64,
    pub temperature: f64,
    pub zero: f64,
    pub wvnum_space: f64,
    pub pressure: f64,
}

impl Header {
    fn new() -> Self {
        Header {
            short_mol_name: String::new(),
            wvnum_start: 0.0,
            wvnum_end: 0.0,
            num_points: 0,
            temperature: 0.0,
            zero: 0.0,
            wvnum_space: 0.0,
            pressure: 0.0,
        }
    }
}

fn read_hitran_header(line: &str) -> Header {
    let short_mol_name = line[10..20].trim().to_string();
    let wvnum_start: f64 = line[20..31].trim().parse().unwrap();
    let wvnum_end: f64 = line[31..42].trim().parse().unwrap();
    let num_points: i64 = line[42..50].trim().parse().unwrap();
    let temperature: f64 = line[50..58].trim().parse().unwrap();
    let zero: f64 = line[59..62].trim().parse().unwrap();
    // let wvnum_space: f64 = line[62..71].trim().parse().unwrap();  // unsure on this one...
    let pressure: f64 = line[71..80].trim().parse().unwrap();

    let wvnum_space = (wvnum_end - wvnum_start) / ((num_points - 1) as f64);

    Header {
        short_mol_name: short_mol_name,
        wvnum_start: wvnum_start,
        wvnum_end: wvnum_end,
        num_points: num_points,
        temperature: temperature,
        zero: zero,
        wvnum_space: wvnum_space,
        pressure: pressure,
    }
}

pub fn read_fwf_xsec(path: PathBuf) -> Option<XsecDatabase> {
    let file = fs::File::open(path).unwrap();
    let reader = BufReader::new(file);

    let mut all_xs: Vec<Array1<f64>> = vec![];
    let mut all_wvnum: Vec<Array1<f64>> = vec![];

    let mut xs: Vec<f64> = vec![];
    let mut wvnum: Vec<f64> = vec![];
    let mut pressure: Vec<f64> = vec![];
    let mut temperature: Vec<f64> = vec![];

    let mut header = Header::new();
    let mut cur_wvnum = 0.0;
    let mut cur_index = 0;

    for line_result in reader.lines() {
        let line = line_result.ok()?;

        // Check if this is header information
        if line.len() == 102 {
            header = read_hitran_header(&line);
            cur_wvnum = header.wvnum_start;
        } else {
            for i in 0..10 {
                let start = i * 10;
                let end = start + 10;
                let value: f64 = line[start..end].trim().parse().unwrap();
                xs.push(value);
                wvnum.push(cur_wvnum);
                cur_wvnum += header.wvnum_space;

                cur_index += 1;
                if cur_index >= header.num_points {
                    all_xs.push(Array1::from(xs.clone()));
                    all_wvnum.push(Array1::from(wvnum.clone()));
                    pressure.push(header.pressure);
                    temperature.push(header.temperature);

                    xs.clear();
                    wvnum.clear();
                    break;
                }
            }
        }
    }

    let mut params = HashMap::new();
    params.insert("pressure".to_string(), pressure);
    params.insert("temperature".to_string(), temperature);

    Some(XsecDatabase {
        xsec: all_xs,
        wvnum: all_wvnum,
        params: params,
    })
}

pub fn read_fwf_folder(folder: PathBuf) -> Option<XsecDatabase> {
    let paths = fs::read_dir(folder).unwrap();

    let mut combined_ds = XsecDatabase {
        xsec: vec![],
        wvnum: vec![],
        params: HashMap::new(),
    };

    for path in paths {
        let path = path.unwrap().path();
        if path.extension().unwrap() == "xsc" {
            let dbase = read_fwf_xsec(path.clone());

            combined_ds += dbase?;
        }
    }

    Some(combined_ds)
}
