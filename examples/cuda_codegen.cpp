#include "math/tiny/tiny_algebra.hpp"
#include "utils/differentiation.hpp"
#include "utils/file_utils.hpp"
#include "utils/stopwatch.hpp"

const std::size_t kInputDim = 2;
const std::size_t kOutputDim = 2;

// function to be converted to CUDA code
template <typename Scalar>
std::vector<Scalar> my_function(const std::vector<Scalar>& v) {
  using std::sqrt;
  std::vector<Scalar> result(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    const auto& e = v[i];
    result[i] = e * e;
    result[i] += tds::where_ge(e, Scalar(0), e * e, e * e * e);
  }
  return result;
}

template <class Base>
class CudaVariableNameGenerator
    : public CppAD::cg::LangCDefaultVariableNameGenerator<Base> {
 protected:
  // defines how many input indices belong to the global input
  std::size_t global_dim_{0};
  // name of thread-local input
  std::string local_name_;

 public:
  inline explicit CudaVariableNameGenerator(
      std::size_t global_dim, std::string depName = "y",
      std::string indepName = "x", std::string localName = "xj",
      std::string tmpName = "v", std::string tmpArrayName = "array",
      std::string tmpSparseArrayName = "sarray")
      : CppAD::cg::LangCDefaultVariableNameGenerator<Base>(
            depName, indepName, tmpName, tmpArrayName, tmpSparseArrayName),
        global_dim_(global_dim),
        local_name_(std::move(localName)) {}

  inline std::string generateIndependent(
      const CppAD::cg::OperationNode<Base>& independent, size_t id) override {
    this->_ss.clear();
    this->_ss.str("");

    if (id - 1 >= global_dim_) {
      // global inputs access directly independent vars starting from index 0
      this->_ss << this->local_name_ << "[" << (id - 1 - global_dim_) << "]";
    } else {
      // thread-local inputs use 'xj' (offset of input 'x')
      this->_ss << this->_indepName << "[" << (id - 1) << "]";
    }

    return this->_ss.str();
  }
};

template <class Base>
class CudaSourceGen : public CppAD::cg::ModelCSourceGen<Base> {
  using CGBase = CppAD::cg::CG<Base>;

  std::size_t global_dim{0};

 public:
  CudaSourceGen(CppAD::ADFun<CppAD::cg::CG<Base>>& fun, std::string model)
      : CppAD::cg::ModelCSourceGen<Base>(fun, model) {}

  const std::map<std::string, std::string>& sources() {
    auto mtt = CppAD::cg::MultiThreadingType::NONE;
    CppAD::cg::JobTimer* timer = nullptr;
    return this->getSources(mtt, timer);
  }

  /**
   * Generate CUDA library code for the forward zero pass.
   */
  std::string zero_source() {
    const std::string jobName = "model (zero-order forward)";

    this->startingJob("'" + jobName + "'", CppAD::cg::JobTimer::GRAPH);

    CppAD::cg::CodeHandler<Base> handler;
    handler.setJobTimer(this->_jobTimer);

    const std::size_t input_dim = this->_fun.Domain();
    const std::size_t output_dim = this->_fun.Range();

    std::cout << "Generating code for function with input dimension "
              << input_dim << " and output dimension " << output_dim << "...\n";

    if (global_dim > input_dim) {
      std::cerr << "CUDA codegen failed: global data input size must not be "
                   "larger than the provided input vector size.\n";
      std::exit(1);
    }

    std::vector<CGBase> indVars(input_dim);
    handler.makeVariables(indVars);
    if (this->_x.size() > 0) {
      for (std::size_t i = 0; i < indVars.size(); i++) {
        indVars[i].setValue(this->_x[i]);
      }
    }

    std::vector<CGBase> dep;

    if (this->_loopTapes.empty()) {
      dep = this->_fun.Forward(0, indVars);
    } else {
      /**
       * Contains loops
       */
      dep = this->prepareForward0WithLoops(handler, indVars);
    }

    this->finishedJob();

    CppAD::cg::LanguageC<Base> langC(this->_baseTypeName);
    langC.setMaxAssignmentsPerFunction(this->_maxAssignPerFunc,
                                       &this->_sources);
    langC.setMaxOperationsPerAssignment(this->_maxOperationsPerAssignment);
    langC.setParameterPrecision(this->_parameterPrecision);
    langC.setGenerateFunction("");  // this->_name + "_forward_zero");

    std::ostringstream code;
    CudaVariableNameGenerator<Base> nameGen(global_dim);

    handler.generateCode(code, langC, dep, nameGen, this->_atomicFunctions,
                         jobName);

    std::size_t temporary_dim = nameGen.getMaxTemporaryVariableID() + 1 -
                                nameGen.getMinTemporaryVariableID();
    std::cout << "Code generated with " << temporary_dim
              << " temporary variables.\n";
    // for (const auto& var : nameGen.getTemporary()) {
    //   std::cout << "\t" << var.name << std::endl;
    // }

    std::ostringstream complete;

    complete << "#include <math.h>\n#include <stdio.h>\n\n";

    complete << "typedef " << this->_baseTypeName << " Float;\n\n";

    complete << "__global__\n";
    std::string kernel_name = std::string(this->_name) + "_forward_zero_kernel";
    std::string fun_head_start = "void " + kernel_name + "(";
    std::string fun_arg_pad = std::string(fun_head_start.size(), ' ');
    complete << fun_head_start;
    complete << "int num_total_threads,\n";
    complete << fun_arg_pad << "Float *output,\n";
    complete << fun_arg_pad << "const Float *input) {\n";
    complete << "   const int i = blockIdx.x * blockDim.x + threadIdx.x;\n";
    complete << "   if (i >= num_total_threads) return;\n";
    complete << "   const int j = " << global_dim << " + i * "
             << (input_dim - global_dim)
             << ";  // thread-local input index offset\n\n";
    complete << "   Float v[" << temporary_dim << "];\n";
    if (global_dim > 0) {
      complete << "   const Float *x = &(input[0]);\n";
    }
    complete << "   const Float *xj = &(input[j]);\n";
    complete << "   Float *y = &(output[i * " << output_dim << "]);\n";

    complete << "\n";

    complete << code.str();

    complete << "}\n\n";

    complete << R"(void allocate(void **x, size_t size) {
  cudaError status = cudaMalloc(x, size);
  if (status != cudaSuccess) {
    fprintf(stderr, "Error %i (%s) while allocating CUDA memory: %s.\n",
    status, cudaGetErrorName(status), cudaGetErrorString(status));
    exit((int)status);
  }
})";

    complete << "\n\n";

    fun_head_start =
        "extern \"C\" void " + std::string(this->_name) + "_forward_zero(";
    fun_arg_pad = std::string(fun_head_start.size(), ' ');
    complete << fun_head_start;
    complete << "int num_total_threads,\n";
    complete << fun_arg_pad << "int num_blocks,\n";
    complete << fun_arg_pad << "int num_threads_per_block,\n";
    complete << fun_arg_pad << "Float *output,\n";
    complete << fun_arg_pad << "const Float *input) {\n";

    complete << "  const size_t output_dim = num_total_threads * " << output_dim
             << ";\n";
    complete << "  const size_t input_dim = num_total_threads * "
             << (input_dim - global_dim) << " + " << global_dim << ";\n\n";

    complete << "  Float* dev_output = nullptr;\n";
    complete << "  Float* dev_input = nullptr;\n\n";

    complete << R"(  allocate((void**)&dev_output, output_dim * sizeof(Float));
  allocate((void**)&dev_input, input_dim * sizeof(Float));

  // Copy input vector from host memory to GPU buffers.
  cudaMemcpy(dev_input, input, input_dim * sizeof(Float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_output, output, output_dim * sizeof(Float), cudaMemcpyHostToDevice);

  )";
    complete << kernel_name;
    complete
        << R"(<<<num_blocks, num_threads_per_block>>>(num_total_threads, dev_output, dev_input);
  
  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaDeviceSynchronize();

  // Copy output vector from GPU buffer to host memory.
  cudaMemcpy(output, dev_output, output_dim * sizeof(Float), cudaMemcpyDeviceToHost);

  cudaFree(dev_output);
  cudaFree(dev_input);
  cudaDeviceReset();)";

    complete << "\n}\n";

    return complete.str();
  }
};

template <typename Scalar>
struct CudaModel {
 protected:
  const std::string model_name_;
  void* lib_handle_;

 public:
  // loads the shared library
  CudaModel(const std::string& model_name, int dlOpenMode = RTLD_NOW)
      : model_name_(model_name) {
    // load the dynamic library
    std::string path = model_name + ".so";
    std::string abs_path;
    bool found = tds::FileUtils::find_file(path, abs_path);
    assert(found);
    lib_handle_ = dlopen(abs_path.c_str(), dlOpenMode);
    // _dynLibHandle = dlmopen(LM_ID_NEWLM, path.c_str(), RTLD_NOW);
    CPPADCG_ASSERT_KNOWN(lib_handle_ != nullptr,
                         ("Failed to dynamically load library '" + model_name +
                          "': " + dlerror())
                             .c_str())
  }

  bool forward_zero(std::vector<std::vector<Scalar>>* thread_outputs,
                    const std::vector<std::vector<Scalar>>& thread_inputs,
                    int num_threads_per_block = 32,
                    const std::vector<Scalar>& global_input = {}) const {
    if (thread_outputs == nullptr || thread_outputs->empty() ||
        (*thread_outputs)[0].empty()) {
      assert(false);
      return false;
    }
    if (thread_outputs->size() != thread_inputs.size()) {
      assert(false);
      return false;
    }

    // signature must match library function
    typedef void (*function_t)(int, int, int, Scalar*, const Scalar*);

    // reset errors
    dlerror();
    std::string function_name = model_name_ + "_forward_zero";
    function_t fun = (function_t)dlsym(lib_handle_, function_name.c_str());
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
      std::cerr << "Cannot load symbol '" << function_name
                << "': " << dlsym_error << '\n';
      dlclose(lib_handle_);
      return false;
    }

    auto num_total_threads = static_cast<int>(thread_inputs.size());
    // concatenate thread-wise inputs and global memory into contiguous input
    // array
    Scalar* input = new Scalar[global_input.size() +
                               thread_inputs[0].size() * num_total_threads];
    std::size_t i = 0;
    for (; i < global_input.size(); ++i) {
      input[i] = global_input[i];
    }
    for (const auto& thread : thread_inputs) {
      for (const Scalar& t : thread) {
        input[i] = t;
        ++i;
      }
    }
    Scalar* output = new Scalar[thread_outputs[0].size() * num_total_threads];

    int num_blocks = ceil(num_total_threads * 1. / num_threads_per_block);

    // call GPU kernel
    fun(num_total_threads, num_blocks, num_threads_per_block, output, input);

    // assign thread-wise outputs
    i = 0;
    for (auto& thread : *thread_outputs) {
      for (Scalar& t : thread) {
        t = output[i];
        ++i;
      }
    }

    delete[] input;
    delete[] output;

    return true;
  }
};

int main(int argc, char* argv[]) {
  using Scalar = double;
  using CGScalar = typename CppAD::cg::CG<Scalar>;
  using Dual = typename CppAD::AD<CGScalar>;

  // trace function with all zeros as input
  std::vector<Dual> ax(kInputDim, Dual(0));
  CppAD::Independent(ax);
  std::vector<Dual> ay(kOutputDim);
  std::cout << "Tracing function for code generation...\n";
  ay = my_function(ax);
  CppAD::ADFun<CGScalar> tape;
  tape.Dependent(ax, ay);
  // tape.optimize();

  tds::Stopwatch timer;
  timer.start();
  std::string model_name = "cuda_model";
  CudaSourceGen<Scalar> cgen(tape, model_name);
  cgen.setCreateForwardZero(true);

  // CppAD::cg::ModelLibraryCSourceGen<Scalar> libcgen(cgen);
  // libcgen.setVerbose(true);
  // CppAD::cg::DynamicModelLibraryProcessor<Scalar> p(libcgen);
  // auto compiler = std::make_unique<CppAD::cg::ClangCompiler<Scalar>>();
  // compiler->setSourcesFolder("cgen_srcs");
  // compiler->setSaveToDiskFirst(true);
  // compiler->addCompileFlag("-O" + std::to_string(1));
  // p.setLibraryName(model_name);
  // p.createDynamicLibrary(*compiler, false);

  // generate CUDA code
  std::string source_zero = cgen.zero_source();
  std::cout << "Zero source:\n" << source_zero << std::endl;
  std::ofstream cuda_file(model_name + ".cu");
  cuda_file << source_zero;
  cuda_file.close();

  // compile shared library
  std::string stdout_msg, stderr_msg;
  CppAD::cg::system::callExecutable(
      "/usr/bin/nvcc",
      {"--ptxas-options=-v", "--compiler-options", "-fPIC", "-o",
       model_name + ".so", "--shared", model_name + ".cu"},
      &stdout_msg, &stderr_msg);
  std::cout << stdout_msg << std::endl;
  std::cerr << stderr_msg << std::endl;

  // create model to load shared library
  CudaModel<Scalar> model(model_name);

  // how many threads to run on the GPU
  int num_total_threads = 20;

  std::vector<std::vector<Scalar>> outputs(num_total_threads,
                                           std::vector<Scalar>(kOutputDim));
  // generate some fake inputs
  std::vector<std::vector<Scalar>> inputs(num_total_threads);
  for (int i = 0; i < num_total_threads; ++i) {
    inputs[i] = std::vector<Scalar>(kInputDim, i);
    inputs[i][1] = -inputs[i][1];
  }

  // call GPU kernel
  model.forward_zero(&outputs, inputs);

  for (const auto& thread : outputs) {
    for (const Scalar& t : thread) {
      std::cout << t << "  ";
    }
    std::cout << std::endl;
  }

  return EXIT_SUCCESS;
}