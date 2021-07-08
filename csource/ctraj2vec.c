#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *method_traj2vec(PyObject *self, PyObject *args){
    PyObject *trajectory = NULL;
    double *frequency = NULL;

    /* Parse arguments */
    if (!PyArg_ParseTuple(args, "Od", &trajectory, &frequency)){
        return NULL;
    }

    /* if (!PyLong_Check(trajectory)){
        PyErr_SetString(PyExc_TypeError, "Must be integer!");
        return (PyObject *) NULL;
    } */

    const char *attr = "mode_list";
    printf(PyObject_HasAttr(trajectory, attr));
    return 1;
}

static PyObject *method_vec2traj(PyObject *self, PyObject *args){
    long *number = NULL;

    /* Parse arguments */
    if (!PyArg_ParseTuple(args, "l", &number)){
        return NULL;
    }

    return PyLong_FromLong(number);
}

static PyMethodDef cTraj2vecMethods[] = {
    {"traj2vec", method_traj2vec, METH_VARARGS, "Python interface for traj2vec C function"},
    {"vec2traj", method_vec2traj, METH_VARARGS, "Python interface for vec2traj C function"},
    {NULL, NULL, 0 , NULL}
};

static struct PyModuleDef ctraj2vecmodule = {
    PyModuleDef_HEAD_INIT,
    "ctraj2vec",
    "Python interface for traj2vec C functions",
    -1,
    cTraj2vecMethods
};

PyMODINIT_FUNC PyInit_ctraj2vec(void){
    return PyModule_Create(&ctraj2vecmodule);
}
