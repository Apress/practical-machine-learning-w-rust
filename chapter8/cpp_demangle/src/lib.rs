use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pymodule]
fn rust2py(py: Python, m: &PyModule) -> PyResult<()> {

    // Note that the `#[pyfn()]` annotation automatically converts the arguments from
    // Python objects to Rust values; and the Rust return value back into a Python object.
    #[pyfn(m, "sum_as_string")]
    fn sum_as_string_py(_py: Python, a:i64, b:i64) -> PyResult<String> {
       Ok(format!("{}", a + b))
    }

    Ok(())
}

#[pyfunction]
fn double(x: usize) -> usize {
    x * 2
}

#[pymodule]
fn module_with_functions(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(double)).unwrap();

    Ok(())
}


/// add(a, b, /)
/// --
///
/// This function adds two unsigned 64-bit integers.
#[pyfunction]
fn add(a: u64, b: u64) -> u64 {
    a + b
}