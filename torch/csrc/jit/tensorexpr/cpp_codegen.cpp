#include <algorithm>
#include <type_traits>
#include <vector>

#include <torch/csrc/jit/tensorexpr/cpp_codegen.h>
#include <torch/csrc/jit/tensorexpr/cpp_intrinsics.h>
#include <torch/csrc/jit/tensorexpr/cpp_tensor.h>
#include <torch/csrc/jit/tensorexpr/cpp_vector.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <torch/csrc/jit/tensorexpr/types.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// Rewrites the variables' name according to valid C++ naming convention.
// E.g. in Graph IR, variable name may contain '.', in C++, they are replaced
// with '_'.
class CppVarNameRewriter : public IRMutator {
 public:
  const Expr* mutate(const Var* v) override {
    constexpr char kDot = '.';
    constexpr char kUnderscore = '_';
    if (v->name_hint().find(kDot) == std::string::npos) {
      return v;
    }
    std::string name = v->name_hint();
    std::replace(name.begin(), name.end(), kDot, kUnderscore);
    const Var* new_v = new Var(name, v->dtype());
    old_to_new_var_[v] = new_v;
    return static_cast<const Expr*>(new_v);
  }

  const Expr* mutate(const Buf* v) override {
    const Var* var = v->base_handle();
    const Var* var_new = static_cast<const Var*>(var->accept_mutator(this));
    if (var_new == var) {
      return v;
    }
    return new Buf(var_new, v->dims(), v->dtype(), v->initializer());
  }

  // After mutation of the TE IR tree, the old Vars may still
  // exist in CodeGen's BufferArgs, this method can be used
  // to map the old Var to the mutated new Var.
  const Var* getNewVar(const Var* old) const {
    if (old_to_new_var_.find(old) != old_to_new_var_.end()) {
      return old_to_new_var_.at(old);
    }
    return old;
  }

 private:
  std::unordered_map<const Var*, const Var*> old_to_new_var_;
};

class ExprVector : public Expr {
 public:
  explicit ExprVector(const std::vector<const Expr*>& exprs)
      : Expr(exprs[0]->dtype()), exprs_(exprs) {}

  size_t size() const {
    return exprs_.size();
  }

  const Expr* operator[](size_t idx) const {
    return exprs_[idx];
  }

  void accept(IRVisitor*) const override {}
  const Expr* accept_mutator(IRMutator*) const override {
    return nullptr;
  }

 private:
  std::vector<const Expr*> exprs_;
};

class StmtVector : public Stmt {
 public:
  explicit StmtVector(const std::vector<const Stmt*>& stmts) : stmts_(stmts) {}

  size_t size() const {
    return stmts_.size();
  }

  const Stmt* operator[](size_t idx) const {
    return stmts_[idx];
  }

  void accept(IRVisitor*) const override {}
  Stmt* accept_mutator(IRMutator*) override {
    return nullptr;
  }

 private:
  std::vector<const Stmt*> stmts_;
};

// Unrolls vector expressions into a vector of scalar expressions.
// For example:
// Ramp(IntImm(0), IntImm(1), IntImm(3)) is unrolled into:
// ExprVector({IntImm(0), Add(IntImm(0), IntImm(1)), Add(IntImm(0),
// Mul(IntImm(2), IntImm(1)))}).
class Devectorizer : public IRMutator {
 public:
#define DEVEC_BINARY_OP(Op)                   \
  const Expr* mutate(const Op* v) override {  \
    auto v1 = devec(v->lhs());                \
    auto v2 = devec(v->rhs());                \
    assert(v1->size() == v2->size());         \
    std::vector<const Expr*> res(v1->size()); \
    for (size_t i = 0; i < res.size(); i++) { \
      res[i] = new Op((*v1)[i], (*v2)[i]);    \
    }                                         \
    return new ExprVector(res);               \
  }
  DEVEC_BINARY_OP(Add)
  DEVEC_BINARY_OP(Sub)
  DEVEC_BINARY_OP(Mul)
  DEVEC_BINARY_OP(Div)
  DEVEC_BINARY_OP(Mod)
  DEVEC_BINARY_OP(Max)
  DEVEC_BINARY_OP(Min)
  DEVEC_BINARY_OP(And)
  DEVEC_BINARY_OP(Or)
  DEVEC_BINARY_OP(Xor)
  DEVEC_BINARY_OP(Lshift)
  DEVEC_BINARY_OP(Rshift)
#undef DEVEC_BINARY_OP

  const Expr* mutate(const CompareSelect* v) override {
    auto lhs = devec(v->lhs());
    auto rhs = devec(v->rhs());
    auto t = devec(v->ret_val1());
    auto f = devec(v->ret_val2());
    assert(lhs->size() == rhs->size());
    assert(t->size() == f->size());
    assert(lhs->size() == t->size());
    std::vector<const Expr*> res(lhs->size());
    for (size_t i = 0; i < res.size(); i++) {
      res[i] = new CompareSelect(
          (*lhs)[i], (*rhs)[i], (*t)[i], (*f)[i], v->compare_select_op());
    }
    return new ExprVector(res);
  }

  const Expr* mutate(const IfThenElse* v) override {
    auto t = devec(v->true_value());
    auto f = devec(v->false_value());
    std::vector<const Expr*> res(
        static_cast<size_t>(v->true_value()->dtype().lanes()));
    for (size_t i = 0; i < res.size(); i++) {
      res[i] = new IfThenElse(v->condition(), (*t)[i], (*f)[i]);
    }
    return new ExprVector(res);
  }

  const Expr* mutate(const Cast* v) override {
    auto src = devec(v->src_value());
    std::vector<const Expr*> res(src->size());
    for (size_t i = 0; i < res.size(); i++) {
      res[i] = new Cast(v->dtype().scalar_dtype(), (*src)[i]);
    }
    return new ExprVector(res);
  }

  const Expr* mutate(const BitCast* v) override {
    auto src = devec(v->src_value());
    std::vector<const Expr*> res(src->size());
    for (size_t i = 0; i < res.size(); i++) {
      res[i] = new BitCast(v->dtype().scalar_dtype(), (*src)[i]);
    }
    return new ExprVector(res);
  }

  const Expr* mutate(const Ramp* v) override {
    std::vector<const Expr*> res(v->lanes());
    for (size_t i = 0; i < res.size(); i++) {
      res[i] = new Add(v->base(), new Mul(new IntImm(i), v->stride()));
    }
    return new ExprVector(res);
  }

  const Expr* mutate(const Broadcast* v) override {
    return new ExprVector(std::vector<const Expr*>(v->lanes(), v->value()));
  }

  const Expr* mutate(const Load* v) override {
    std::vector<const ExprVector*> devec_indices(v->indices().size());
    for (size_t i = 0; i < devec_indices.size(); i++) {
      devec_indices[i] = devec(v->indices()[i]);
    }
    auto devec_mask = devec(v->mask());
    std::vector<const Expr*> res(devec_indices[0]->size());
    std::vector<const Expr*> indices(v->indices().size());
    for (size_t i = 0; i < res.size(); i++) {
      for (size_t j = 0; j < indices.size(); j++) {
        indices[j] = (*devec_indices[j])[i];
      }
      res[i] = new Load(
          v->dtype().scalar_dtype(), v->buf(), indices, (*devec_mask)[i]);
    }
    return new ExprVector(res);
  }

  Stmt* mutate(const Store* v) override {
    std::vector<const ExprVector*> devec_indices(v->indices().size());
    for (size_t i = 0; i < devec_indices.size(); i++) {
      devec_indices[i] = devec(v->indices()[i]);
    }
    auto devec_mask = devec(v->mask());
    auto devec_val = devec(v->value());
    std::vector<const Stmt*> res(devec_indices[0]->size());
    std::vector<const Expr*> indices(v->indices().size());
    for (size_t i = 0; i < res.size(); i++) {
      for (size_t j = 0; j < indices.size(); j++) {
        indices[j] = (*devec_indices[j])[i];
      }
      res[i] = new Store(v->buf(), indices, (*devec_val)[i], (*devec_mask)[i]);
    }
    return new StmtVector(res);
  }

  const Expr* mutate(const Intrinsics* v) override {
    if (v->nparams() == 0) {
      return new ExprVector({v});
    }

    std::vector<const ExprVector*> devec_params(v->nparams());
    for (size_t i = 0; i < devec_params.size(); i++) {
      devec_params[i] = devec(v->param(static_cast<int>(i)));
    }

    std::vector<const Expr*> res(v->param(0)->dtype().lanes());
    std::vector<const Expr*> params(v->nparams());
    for (size_t i = 0; i < res.size(); i++) {
      for (size_t p = 0; p < params.size(); p++) {
        params[p] = (*devec_params[p])[i];
      }
      res[i] = new Intrinsics(v->op_type(), params);
    }
    return new ExprVector(res);
  }

  const Expr* mutate(const Var* v) override {
    if (vector_vars_.find(v) == vector_vars_.end()) {
      throw std::runtime_error("Var is not of vector data type");
    }
    return vector_vars_.at(v);
  }

  void devecVar(const Var* v) {
    vector_vars_[v] = devec(v);
  }

  const ExprVector* devec(const Expr* v) {
    return static_cast<const ExprVector*>(v->accept_mutator(this));
  }

  const StmtVector* devec(const Stmt* v) {
    return static_cast<const StmtVector*>(
        const_cast<Stmt*>(v)->accept_mutator(this));
  }

 private:
  std::unordered_map<const Var*, const ExprVector*> vector_vars_;
};

CppPrinter::CppPrinter(std::ostream* os)
    : IRPrinter(*os), devectorizer_(std::make_unique<Devectorizer>()) {}

CppPrinter::~CppPrinter() = default;

void CppPrinter::printPrologue() {
  os() << "#include <cassert>" << std::endl;
  os() << "#include <cmath>" << std::endl;
  os() << "#include <vector>" << std::endl;
  os() << "#include <algorithm>" << std::endl;
  os() << "#include <type_traits>" << std::endl;
  os() << std::endl;

  os() << "#define POS_INFINITY INFINITY" << std::endl;
  os() << "#define NEG_INFINITY -INFINITY" << std::endl;
  os() << std::endl;

  os() << cpp_intrinsics_definition << std::endl;
  os() << std::endl;

  os() << "namespace torch {" << std::endl;
  os() << "namespace jit {" << std::endl;
  os() << "namespace tensorexpr {" << std::endl;
  for (auto const& it : getNNCFunctionRegistry()) {
    os() << declareExternalFunction(it.first) << std::endl;
  }
  os() << "} // namespace tensorexpr" << std::endl;
  os() << "} // namespace jit" << std::endl;
  os() << "} // namespace torch" << std::endl;
  os() << std::endl;

  os() << "using namespace torch::jit::tensorexpr;" << std::endl;
  os() << std::endl;
}

std::string CppPrinter::declareExternalFunction(const std::string& func_name) {
  return "void " + func_name +
      "("
      "int64_t bufs_num, "
      "void** buf_data, "
      "int64_t* buf_ranks, "
      "int64_t* buf_dims, "
      "int8_t* buf_dtypes, "
      "int64_t args_num, "
      "int64_t* extra_args);";
}

template <typename T>
inline typename std::enable_if<!std::is_floating_point<T>::value, void>::type
visit_mod(std::ostream& os, const Expr* lhs, const Expr* rhs) {
  os << *lhs << " % " << *rhs;
}

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, void>::type
visit_mod(std::ostream& os, const Expr* lhs, const Expr* rhs) {
  os << "std::fmod(" << *lhs << ", " << *rhs << ")";
}

template <typename T>
inline typename std::enable_if<
    std::is_floating_point<T>::value || std::is_integral<T>::value,
    void>::type
visit_max(std::ostream& os, const Expr* lhs, const Expr* rhs) {
  os << "std::max(" << *lhs << ", " << *rhs << ")";
}

template <typename T>
inline typename std::enable_if<
    !std::is_floating_point<T>::value && !std::is_integral<T>::value,
    void>::type
visit_max(std::ostream& os, const Expr* lhs, const Expr* rhs) {
  os << "(" << *lhs << " < " << *rhs << ") ? " << *rhs << " : " << *lhs;
}

template <typename T>
inline typename std::enable_if<
    std::is_floating_point<T>::value || std::is_integral<T>::value,
    void>::type
visit_min(std::ostream& os, const Expr* lhs, const Expr* rhs) {
  os << "std::min(" << *lhs << ", " << *rhs << ")";
}

template <typename T>
inline typename std::enable_if<
    !std::is_floating_point<T>::value && !std::is_integral<T>::value,
    void>::type
visit_min(std::ostream& os, const Expr* lhs, const Expr* rhs) {
  os << *lhs << " < " << *rhs << " ? " << *lhs << " : " << *rhs;
}

template <typename T>
void visit_binary_op(
    std::ostream& os,
    const Expr* lhs,
    const Expr* rhs,
    IRNodeType op_type) {
  switch (op_type) {
    case IRNodeType::kMod:
      visit_mod<T>(os, lhs, rhs);
      break;
    case IRNodeType::kMax:
      visit_max<T>(os, lhs, rhs);
      break;
    case IRNodeType::kMin:
      visit_min<T>(os, lhs, rhs);
      break;
    default:
      throw std::runtime_error("invalid op type");
  }
}

template <typename Op>
void dispatch_binary_op(std::ostream& os, const BinaryOpNode<Op>* v) {
  switch (v->lhs()->dtype().scalar_type()) {
#define TYPE_CASE(Type, Name)                                      \
  case ScalarType::Name:                                           \
    visit_binary_op<Type>(os, v->lhs(), v->rhs(), v->expr_type()); \
    break;
    AT_FORALL_SCALAR_TYPES_AND2(Half, Bool, TYPE_CASE);
#undef TYPE_CASE
    default:
      throw unsupported_dtype();
  }
}

void CppPrinter::visit(const Ramp* v) {
  throw unimplemented_lowering(v);
}

void CppPrinter::visit(const Broadcast* v) {
  throw unimplemented_lowering(v);
}

void CppPrinter::visit(const Mod* v) {
  dispatch_binary_op(os(), v);
}

void CppPrinter::visit(const Max* v) {
  dispatch_binary_op(os(), v);
}

void CppPrinter::visit(const Min* v) {
  dispatch_binary_op(os(), v);
}

void CppPrinter::visit(const CompareSelect* v) {
  os() << "((" << *v->lhs() << " "
       << IRPrinter::to_string(v->compare_select_op()) << " " << *v->rhs()
       << ") ? " << *v->ret_val1() << " : " << *v->ret_val2() << ")";
}

void CppPrinter::visit(const IfThenElse* v) {
  os() << "((" << *v->condition() << ") ? " << *v->true_value() << " : "
       << *v->false_value() << ")";
}

void CppPrinter::visit(const Allocate* v) {
  size_t size = v->dtype().byte_size();
  for (auto dim : v->dims()) {
    const IntImm* d = dynamic_cast<const IntImm*>(dim);
    if (d) {
      size *= d->value();
    } else {
      throw std::runtime_error("Only IntImm dimensions are supported for now");
    }
  }

  emitIndent();
  os() << v->dtype().ToCppString() << "* " << (*v->buffer_var())
       << " = static_cast<" << v->dtype().ToCppString() << "*>(malloc(" << size
       << "));" << std::endl;
}

void CppPrinter::visit(const Free* v) {
  emitIndent();
  os() << "free(" << *v->buffer_var() << ");" << std::endl;
}

void CppPrinter::visit(const Load* v) {
  auto flat_idx = flatten_index(v->buf()->dims(), v->indices());
  const IntImm* m = dynamic_cast<const IntImm*>(v->mask());
  if (m == nullptr) {
    os() << "((" << *v->mask() << ") ? " << *v->base_handle() << "["
         << *flat_idx << "] : 0)";
  } else if (m->value() == 0) {
    os() << "0";
  } else {
    os() << *v->base_handle() << "[" << *flat_idx << "]";
  }
}

void CppPrinter::visit(const Store* v) {
  if (v->value()->dtype().lanes() > 1) {
    const StmtVector* stores = devectorizer_->devec(v);
    for (size_t i = 0; i < stores->size(); i++) {
      visit(static_cast<const Store*>((*stores)[i]));
    }
  } else {
    auto flat_idx = flatten_index(v->buf()->dims(), v->indices());
    emitIndent();
    const IntImm* m = dynamic_cast<const IntImm*>(v->mask());
    if (m == nullptr) {
      os() << "if (" << *v->mask() << ") {" << std::endl;
      indent_++;
      emitIndent();
      os() << *v->base_handle() << "[" << *flat_idx << "] = " << *v->value()
           << ";" << std::endl;
      indent_--;
      emitIndent();
      os() << "}" << std::endl;
    } else if (m->value() != 0) {
      os() << *v->base_handle() << "[" << *flat_idx << "] = " << *v->value()
           << ";" << std::endl;
    }
  }
}

void CppPrinter::visit(const Cast* v) {
  os() << "static_cast<" << v->dtype().ToCppString() << ">(" << *v->src_value()
       << ")";
}

void CppPrinter::visit(const BitCast* v) {
  os() << "std::bitcast<" << v->src_value()->dtype().ToCppString() << ", "
       << v->dtype().ToCppString() << ">(" << *v->src_value() << ")";
}

void CppPrinter::visit(const Intrinsics* v) {
  if (v->op_type() == kRand || v->op_type() == kSigmoid) {
    throw std::runtime_error("kRand and kSigmoid are not supported");
  }

  os() << "std::" << v->func_name() << "(";
  for (int i = 0; i < v->nparams(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    os() << *v->param(i);
  }
  os() << ")";
}

void CppPrinter::visit(const ExternalCall* v) {
  // The generated code needs to link against functions defined
  // in external_functions.cpp.

  auto& func_registry = getNNCFunctionRegistry();
  if (!func_registry.count(v->func_name())) {
    throw unimplemented_lowering(v);
  }

  std::vector<const Buf*> bufs(v->buf_args());
  bufs.insert(bufs.begin(), v->buf());
  auto for_buf = [&](std::function<void(const Buf*)> print_buf) {
    for (size_t i = 0; i < bufs.size(); i++) {
      if (i > 0) {
        os() << ", ";
      }
      print_buf(bufs[i]);
    }
  };

  emitIndent();
  os() << "{" << std::endl;
  indent_++;

  emitIndent();
  os() << "std::vector<void*> buf_ptrs{";
  for_buf([&](const Buf* b) { os() << *b->base_handle(); });
  os() << "};" << std::endl;

  emitIndent();
  os() << "std::vector<int64_t> buf_ranks{";
  for_buf([&](const Buf* b) { os() << b->ndim(); });
  os() << "};" << std::endl;

  emitIndent();
  os() << "std::vector<int64_t> buf_dims{";
  for_buf([&](const Buf* buf) {
    for (size_t i = 0; i < buf->ndim(); i++) {
      if (i > 0) {
        os() << ", ";
      }
      os() << *buf->dim(i);
    }
  });
  os() << "};" << std::endl;

  emitIndent();
  os() << "std::vector<int8_t> buf_dtypes{";
  for_buf([&](const Buf* buf) {
    os() << static_cast<int>(buf->dtype().scalar_type());
  });
  os() << "};" << std::endl;

  emitIndent();
  os() << "std::vector<int64_t> extra_args{";
  for (size_t i = 0; i < v->args().size(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    os() << *v->args()[i];
  }
  os() << "};" << std::endl;

  emitIndent();
  os() << v->func_name() << "(" << std::endl;
  emitIndent();
  os() << "    buf_ptrs.size()," << std::endl;
  emitIndent();
  os() << "    buf_ptrs.data()," << std::endl;
  emitIndent();
  os() << "    buf_ranks.data()," << std::endl;
  emitIndent();
  os() << "    buf_dims.data()," << std::endl;
  emitIndent();
  os() << "    buf_dtypes.data()," << std::endl;
  emitIndent();
  os() << "    extra_args.size()," << std::endl;
  emitIndent();
  os() << "    extra_args.data());" << std::endl;

  indent_--;
  emitIndent();
  os() << "}" << std::endl;
}

void CppPrinter::visit(const Let* v) {
  if (v->dtype().lanes() == 1) {
    emitIndent();
    os() << v->dtype().ToCppString() << " " << *v->var() << " = " << *v->value()
         << ";" << std::endl;
  } else {
    devectorizer_->devecVar(v->var());
  }
}

CppCodeGen::CppCodeGen(
    Stmt* stmt,
    const std::vector<BufferArg>& buffer_args,
    at::Device device,
    const std::string& kernel_func_name)
    : CodeGen(stmt, buffer_args, device, kernel_func_name) {
  init();
}

void CppCodeGen::init() {
  printer_ = std::make_unique<CppPrinter>(&oss_);
  var_name_rewriter_ = std::make_unique<CppVarNameRewriter>();

  apply_mutator(var_name_rewriter_.get());

  printer_->printPrologue();
  os() << "void " << kernel_func_name() << "(";
  const std::vector<BufferArg> buffer_args = this->buffer_args();
  for (size_t i = 0; i < buffer_args.size(); i++) {
    if (i > 0) {
      os() << ", ";
    }
    const BufferArg& buffer_arg = buffer_args[i];
    const Var* var = var_name_rewriter_->getNewVar(buffer_arg.var());
    Dtype dtype = buffer_arg.dtype();
    os() << dtype.ToCppString() << (buffer_arg.isVar() ? " " : "* ") << *var;
  }
  os() << ")";
  stmt()->accept(printer_.get());
  os() << std::endl;
}

CppCodeGen::~CppCodeGen() = default;

void CppCodeGen::call(const std::vector<CallArg>& args) {
  // TODO: compile the generated C++ kernel into a library,
  // and call the library here.
  os() << "int main() {}" << std::endl;
}

RegisterCodeGen<CppCodeGen> cpp_codegen_reg("cpp_codegen");

} // namespace tensorexpr
} // namespace jit
} // namespace torch
