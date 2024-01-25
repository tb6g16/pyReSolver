from setuptools import setup, find_packages

def main():
    setup(
        name = 'pyReSolver',
        packages = find_packages(include = ['pyReSolver', 'pyReSolver.*']),
        install_requires = ['numpy',
                            'scipy',
                            'matplotlib',
                            'h5py',
                            'pyfftw']
    )

if __name__ == '__main__':
    main()
