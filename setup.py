from distutils.core import setup, Extension

def main():
    setup(name = "ctrajmul",
          description = "Python interface for trajmul C function",
          author = "Thomas Burton",
          author_email = "tb6g16@soton.ac.uk",
          ext_modules = [Extension("ctrajmul", ["./ResolventSolver/csrc/ctrajmul.c"])])

# ext_package = 'ResolventSolver'

if __name__ == '__main__':
    main()
