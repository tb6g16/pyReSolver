from distutils.core import setup, Extension

def main():
    setup(name = "ctraj2vec",
          version = "0.1",
          description = "Python interface for traj2vec C function",
          author = "Thomas Burton",
          author_email = "tb6g16@soton.ac.uk",
          ext_modules = [Extension("ctraj2vec", ["ctraj2vec.c"])])

if __name__ == '__main__':
    main()
