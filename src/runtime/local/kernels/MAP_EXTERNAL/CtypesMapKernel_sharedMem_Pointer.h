#ifndef SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_CTYPESMAPKERNEL_SHAREDPOINTER_H
#define SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_CTYPESMAPKERNEL_SHAREDPOINTER_H

#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/MAP_EXTERNAL/MapKernelUtils.h>
#include <Python.h>
#include <memory>
#include <util/PythonInterpreter.h>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************
template<typename DTRes, typename DTArg>
struct CtypesMapKernel_SharedMem_Pointer
{
    static void apply(DTRes *& res, const DTArg * arg, const char* func, const char* varName) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************
template<class DTRes, class DTArg>
void ctypesMapKernel_SharedMem_Pointer(DTRes *& res, const DTArg * arg, const char* func, const char* varName) {
    CtypesMapKernel_SharedMem_Pointer<DTRes,DTArg>::apply(res, arg, func, varName);
}

// ----------------------------------------------------------------------------
// DenseMatrix
// ----------------------------------------------------------------------------
template<typename VTRes, typename VTArg>
struct CtypesMapKernel_SharedMem_Pointer<DenseMatrix<VTRes>, DenseMatrix<VTArg>> {

    static void apply(DenseMatrix<VTRes> *& res, const DenseMatrix<VTArg> * arg, const char* func, const char* varName)
    {
        PythonInterpreter::getInstance();

        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        PyObject* pName = PyUnicode_DecodeFSDefault("CtypesMapKernel_sharedMem_Pointer");
        PyObject* pModule = PyImport_Import(pName);
        Py_XDECREF(pName);

        if (!pModule) {
            std::cerr << "Failed to import Python module!" << std::endl;
            PyErr_Print();
            PyGILState_Release(gstate);
            return;
        }

        PyObject* pFunc = PyObject_GetAttrString(pModule, "apply_map_function");
        Py_XDECREF(pModule);

        if (!PyCallable_Check(pFunc)) {
            Py_XDECREF(pFunc);
            PyGILState_Release(gstate);
            std::cerr << "Function not callable!" << std::endl;
            return;
        }

        auto res_sp = res->getValuesSharedPtr();
        auto arg_sp = arg->getValuesSharedPtr();

        std::string orig_dtype_arg = get_dtype_name<VTArg>();
        std::string orig_dtype_res = orig_dtype_arg; // Assuming VTArg and VTRes have the same data type

        PyObject* pArgs = Py_BuildValue("LLiissss",
                                        reinterpret_cast<int64_t>(res_sp.get()),
                                        reinterpret_cast<int64_t>(arg_sp.get()),
                                        res->getNumRows(),
                                        res->getNumCols(),
                                        func,
                                        varName,
                                        orig_dtype_arg.c_str(),
                                        orig_dtype_res.c_str()
                                        );

        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        Py_XDECREF(pFunc);
        Py_XDECREF(pArgs);

        if (!pResult) {
            PyErr_Print();
             PyGILState_Release(gstate);
        } else {
            Py_XDECREF(pResult);
        }

        PyGILState_Release(gstate);

    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_MAP_EXTERNAL_CTYPESMAPKERNEL_SHAREDPOINTER_H