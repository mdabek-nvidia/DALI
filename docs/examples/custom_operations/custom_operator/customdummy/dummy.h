#ifndef EXAMPLE_DUMMY_H_
#define EXAMPLE_DUMMY_H_

#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace other_ns {

template <typename Backend>
class Dummy : public ::dali::Operator<Backend> {
 public:
  inline explicit Dummy(const ::dali::OpSpec &spec) :
    ::dali::Operator<Backend>(spec) {}

  virtual inline ~Dummy() = default;

  Dummy(const Dummy&) = delete;
  Dummy& operator=(const Dummy&) = delete;
  Dummy(Dummy&&) = delete;
  Dummy& operator=(Dummy&&) = delete;

 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc> &output_desc,
                 const ::dali::Workspace &ws) override {
    const auto &input = ws.Input<Backend>(0);
    output_desc.resize(1);
    output_desc[0] = {input.shape(), input.type()};
    return true;
  }

  void RunImpl(::dali::Workspace &ws) override;
};

}  // namespace other_ns

#endif  // EXAMPLE_DUMMY_H_
