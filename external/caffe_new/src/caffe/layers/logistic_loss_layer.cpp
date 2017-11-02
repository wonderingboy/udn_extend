#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype LogLoss(Dtype x, Dtype y) {
  return log(1 + exp(-y*x));
}

template <typename Dtype>
inline Dtype LogLossBack(Dtype x, Dtype y) {
  return -y*exp(-y*x)/(1 + exp(-y*x));
}


template <typename Dtype>
void LogisticLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top); 
}

template <typename Dtype>
void LogisticLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "LOGISTIC Loss layer inputs must have the same count.";
}

template <typename Dtype>
void LogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  //const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    loss += LogLoss(input_data[i], target[i]); 
  }
  top[0]->mutable_cpu_data()[0] = loss / count;
}

template <typename Dtype>
void LogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
	const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	for (int i = 0; i < count; ++i) {
	  bottom_diff[i] = LogLossBack(input_data[i], target[i]); 
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

//#ifdef CPU_ONLY
//STUB_GPU_BACKWARD(LogisticLossLayer, Backward);
//#endif

INSTANTIATE_CLASS(LogisticLossLayer);
REGISTER_LAYER_CLASS(LogisticLoss);

}  // namespace caffe
