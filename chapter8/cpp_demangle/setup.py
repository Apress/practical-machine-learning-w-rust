from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(name='cpp-demangle',
      version="0.0.1",
      rust_extensions=[RustExtension('cpp_demangle', 'Cargo.toml',  binding=Binding.PyO3)],
      test_suite="tests",
      zip_safe=False)
