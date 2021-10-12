from setuptools import setup, find_packages

# THIS STUFF HAS TO IMPLEMENTED FOR THE C EXTENSIONS
# def main():
#     setup(name = "ctrajmul",
#           description = "Python interface for trajmul C function",
#           author = "Thomas Burton",
#           author_email = "tb6g16@soton.ac.uk",
#           ext_modules = [Extension("ctrajmul", ["./ResolventSolver/csrc/ctrajmul.c"])])

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

# ext_package = 'ResolventSolver'

if __name__ == '__main__':
    main()
