#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include </home/tb6g16/Documents/ResolventSolver/venv/lib/python3.6/site-packages/numpy/core/include/numpy/arrayobject.h>

/* define function */
void *method_trajmul(PyObject *self){
    /* initialise variables */
    /* PyObject *arg1 = NULL, *arg2 = NULL; */
    /* PyArrayObject *arr1 = NULL, *arr2 = NULL; */

    /* parse argument tuple */
    /* if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)){
        return NULL;
    } */

    /* convert argument to ndarray */
    /* arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr1 == NULL) return NULL;
    arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr2 == NULL) return NULL; */

    /* printf(PyArray_NDIM(arr1));
    printf(PyArray_NDIM(arr2));
    printf(PyArray_TYPE(arr1));
    printf(PyArray_TYPE(arr2));
    printf(PyArray_DIMS(arr1));
    printf(PyArray_DIMS(arr2));
    printf(PyArray_DESCR(arr1));
    printf(PyArray_DESCR(arr2)); */

    /* Py_DECREF(arr1);
    Py_DECREF(arr2); */
    printf("Hello world!");
}

/* function list */
static PyMethodDef cTrajMulMethods[] = {
    {"trajmul", method_trajmul, METH_VARARGS, "Extension of numpy matmul function with multiplication by first element only"},
    {NULL, NULL, 0, NULL} /* Sentinal */
};

/* module definition */
static struct PyModuleDef ctrajmulmodule = {
    PyModuleDef_HEAD_INIT,
    "ctrajmul",
    "Python interface for trajectory multiplication",
    -1,
    cTrajMulMethods
};

/* module initialisation */
PyMODINIT_FUNC PyInit_ctrajmul(void){
    return PyModule_Create(&ctrajmulmodule);
}
