#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include "decoding.c"
#include "encoding.c"
#include <numpy/ndarraytypes.h>

static PyObject* npy_decoding(PyObject* self, PyObject* args) {
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) {
        return NULL;
    }

    // Ensure array is contiguous and of correct type
    array = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)array, NPY_UINT8, 1, 1);
    if (!array) {
        return NULL;
    }

    // get length and arraydata (elements) from input nparray
    npy_intp length = PyArray_SIZE(array);
    uint8_t* data = (uint8_t*)PyArray_DATA(array);

    // encode/compress input array
    struct decArrayTuple arrayTuple = decompress(data, length);

    // create new array to return
    npy_intp dims = arrayTuple.len;
    // get memory allocated for pointer to return object. &array is random, to get size of a pointer
    PyObject* newNPYArray = (PyObject* ) malloc(sizeof(&array));
    // create new np array object with decoded data
    newNPYArray = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_UINT16), 1, &dims, NULL, arrayTuple.array, 0, NULL);

    Py_DECREF(array);
    Py_INCREF(newNPYArray);
    return newNPYArray;
}

static PyObject* npy_encoding(PyObject* self, PyObject* args) {
    int bitlen;
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &array, &bitlen)) {
        return NULL;
    }

    // Ensure array is contiguous and of correct type
    array = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)array, NPY_UINT16, 1, 1);
    if (!array) {
        return NULL;
    }

    // get length and arraydata (elements) from input nparray
    npy_intp length = PyArray_SIZE(array);
    uint16_t* data = (uint16_t*)PyArray_DATA(array);

    // encode/compress input array
    struct encArrayTuple arrayTuple = compress(bitlen, data, length);

    npy_intp dims = arrayTuple.len;
    // get memory allocated for pointer to return object. &array is random, to get size of a pointer
    PyObject* newNPYArray = (PyObject* ) malloc(sizeof(&array));
    // create new np array object with encoded data
    newNPYArray = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_UINT8), 1, &dims, NULL, arrayTuple.array, 0, NULL);

    Py_DECREF(array); // macht manchmal probleme
    Py_INCREF(newNPYArray);
    return newNPYArray;
}

static PyMethodDef methods[] = {
    {"npy_encoding", npy_encoding, METH_VARARGS, "Encode elements of a NumPy array with variable length encoding"},
    {"npy_decoding", npy_decoding, METH_VARARGS, "Decode elements of a NumPy array with variable length decoding"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cccp_vle",
    "Module for NumPy variable length (vle) encoding and decoding",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_cccp_vle(void) {
    PyObject* module = PyModule_Create(&moduledef);
    if (!module) return NULL;
    import_array(); // Initialize NumPy C-API
    return module;
}