from setuptools import setup, Extension
import pathlib
from Cython.Build import cythonize

here = pathlib.Path(__file__).parent.resolve()
cython_config = {
    'compiler_directives': {
        'language_level': '3'
    },
}

pkg = 'tphenotype'
sources = [r'C:\Users\Linhai Xie\OneDrive\PDPM\ClinicalDecisionMaking\SepsisClustering\models\tphenotype\utils\lexsort.pyx']
extension = Extension('lexsort', sources=sources, extra_compile_args=["-std=c++11"])

extensions = [extension]

setup(
    ext_package=r'C:\Users\Linhai Xie\OneDrive\PDPM\ClinicalDecisionMaking\SepsisClustering\models\tphenotype\utils',
    ext_modules=cythonize(extensions, **cython_config),
)
