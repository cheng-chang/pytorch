#pragma once

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// Rewrites the variables' name according to valid C++ naming convention.
class CppVarNameRewriter : public IRMutator {
 public:
  const Expr* mutate(const Var* v) override;
  const Expr* mutate(const Buf* v) override;

  const Var* getNewVar(const Var* old) const {
    if (old_to_new_var_.find(old) != old_to_new_var_.end()) {
      return old_to_new_var_.at(old);
    }
    return old;
  }

 private:
  std::unordered_map<const Var*, const Var*> old_to_new_var_;
};

// Generates C++ code from the IR.
//
// The generated C++ code relies on:
// 1. Vector defined in cpp_vector.h;
// 2. Tensor defined in cpp_tensor.h.
class TORCH_API CppPrinter : public IRPrinter {
 public:
  explicit CppPrinter(std::ostream* os) : IRPrinter(*os) {}

  void printPrologue();

  using IRPrinter::visit;

  // Vector data types.
  void visit(const Ramp*) override;
  void visit(const Broadcast*) override;

  // Binary expressions.
  void visit(const Mod*) override;
  void visit(const Max*) override;
  void visit(const Min*) override;

  // Conditional expressions.
  void visit(const CompareSelect*) override;
  void visit(const IfThenElse*) override;

  // Tensor operations.
  void visit(const Allocate*) override;
  void visit(const Free*) override;
  void visit(const Load*) override;
  void visit(const Store*) override;

  // Casts.
  void visit(const Cast*) override;
  void visit(const BitCast*) override;

  // Calls.
  void visit(const Intrinsics*) override;
  void visit(const ExternalCall*) override;

 private:
  std::string toLambda(CompareSelectOperation op, const std::string& ty);
  std::string declareExternalFunction(const std::string& func_name);
};

class TORCH_API CppCodeGen : public CodeGen {
 public:
  template <typename... Ts>
  CppCodeGen(Stmt* stmt, Ts... ts)
      : CodeGen(stmt, std::vector<BufferArg>({BufferArg(ts)...}), at::kCPU) {
    init();
  }

  CppCodeGen(
      Stmt* stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::kCPU,
      const std::string& kernel_func_name = "func")
      : CodeGen(stmt, buffer_args, device, kernel_func_name) {
    init();
  }

  void call(const std::vector<CallArg>& args) override;

  template <typename... Ts>
  void operator()(const Ts&... ts) {
    call(std::vector<CallArg>({CallArg(ts)...}));
  }

  std::string getCodeText() override {
    return oss_.str();
  }

 private:
  void init();

  std::ostream& os() {
    return printer_->os();
  }

  std::ostringstream oss_;
  std::unique_ptr<CppPrinter> printer_;
  std::unique_ptr<CppVarNameRewriter> var_name_rewriter_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
