#include <stdio.h>

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "lib.h"


PyObject* fl_init(PyObject* _self, PyObject* args) {
    size_t chunk_size, chunks_per_batch;
    char* file_path;

    if (!PyArg_ParseTuple(args, "nns", &chunk_size, &chunks_per_batch, &file_path)) {
        return NULL;
    }

    // open file
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        PyErr_SetString(PyExc_IOError, "Could not open file");
        return NULL;
    }

    // initialize file loader
    if (init(chunk_size, chunks_per_batch, file) != 0) {
        fclose(file);
        PyErr_SetString(PyExc_IOError, "Could not initialize file loader (OOM probably)");
        return NULL;
    }

    Py_RETURN_NONE;
}

PyObject* fl_get_batch(PyObject* _self, PyObject* _args) {
    const batch_t batch = get_batch();

    // check for EOF batch
    if (batch.num_chunks == 0) {
        Py_RETURN_NONE;
    }

    // wrap in numpy arrays
    const size_t inputs_dims[] = { batch.num_chunks, INPUT_CHUNK_SIZE };
    PyObject* inputs_np_array = PyArray_SimpleNewFromData(2, (npy_intp*) &inputs_dims, NPY_FLOAT32, batch.inputs);

    const size_t targets_dims[] = { batch.num_chunks, TARGET_CHUNK_SIZE };
    PyObject* targets_np_array = PyArray_SimpleNewFromData(2, (npy_intp*) &targets_dims, NPY_FLOAT32, batch.targets);

    // return
    return Py_BuildValue("OO", inputs_np_array, targets_np_array);
}


// -----------------------------
// | Python module declaration |
// -----------------------------
static PyMethodDef fl_methods[] = {
    {"init",  fl_init, METH_VARARGS, "Initialize the file loader. \nArgs: (chunk_size, chunks_per_batch, path_to_file)"},
    {"get_batch", fl_get_batch, METH_NOARGS, "Get a batch from the file loader. Args: None"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyModuleDef fl_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "c_file_loader",
    .m_size = 0,  // non-negative
    .m_methods = fl_methods,
};

PyMODINIT_FUNC
PyInit_ccpc_file_loader(void)
{
    import_array();

    return PyModuleDef_Init(&fl_module);
}
