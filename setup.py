from setuptools import setup, find_packages

def main():
    setup(
        name = 'ResolventSolver',
        packages = find_packages(include = ['ResolventSolver', 'ResolventSolver.*']),
        install_requires = ['numpy',
                            'scipy',
                            'matplotlib',
                            'h5py',
                            'snakeviz']
    )

if __name__ == '__main__':
    main()
