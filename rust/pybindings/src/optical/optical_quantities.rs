use numpy::*;
use pyo3::prelude::*;
use sk_core::optical::{OpticalQuantities, legendre::LegendreAccess};

#[pyclass]
pub struct PyOpticalQuantities {
    oq: OpticalQuantities,
}

impl PyOpticalQuantities {
    pub fn new(oq: OpticalQuantities) -> Self {
        Self { oq }
    }
}

#[pymethods]
impl PyOpticalQuantities {
    #[new]
    fn py_new(num_geometry: usize, num_wavelengths: usize) -> Self {
        let oq = OpticalQuantities::new(num_geometry, num_wavelengths);
        Self { oq }
    }

    #[getter]
    fn get_cross_section<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        let array = &this.borrow().oq.cross_section;

        unsafe { PyArray2::borrow_from_array(array, this.into_any()) }
    }

    #[getter]
    // This is just for interfacing
    fn get_extinction<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        let array = &this.borrow().oq.cross_section;

        unsafe { PyArray2::borrow_from_array(array, this.into_any()) }
    }

    #[getter]
    fn get_ssa<'py>(this: Bound<'py, Self>) -> Bound<'py, PyArray2<f64>> {
        let array = &this.borrow().oq.ssa;

        unsafe { PyArray2::borrow_from_array(array, this.into_any()) }
    }

    #[getter]
    fn get_a1<'py>(this: Bound<'py, Self>) -> Option<Bound<'py, PyArray3<f64>>> {
        let binding = this.borrow();
        let view = binding.oq.legendre.as_ref();

        if let Some(view) = view {
            let a1 = view.view();
            let a1 = LegendreAccess::<Ix3, 4>::new(a1).a1;

            return unsafe { Some(PyArray3::borrow_from_array(&a1, this.into_any())) };
        } else {
            return None;
        }
    }
}
