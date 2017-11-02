#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanPartLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  mask_diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanPartLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  Dtype positives_num=1;
  const Dtype* myMask = bottom[2]->cpu_data();
  for (int iii = 0; iii< count; iii++)
  {
    if (myMask[iii]>0)
    {
      positives_num = positives_num + 1;
    }
  }
  caffe_mul(count, bottom[2]->cpu_data(), bottom[0]->cpu_data(), bottom[0]->mutable_cpu_data());
  caffe_mul(count, bottom[2]->cpu_data(), bottom[1]->cpu_data(), bottom[1]->mutable_cpu_data());
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  //caffe_mul(count, bottom[2]->cpu_data(), diff_.cpu_data(), mask_diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / positives_num;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanPartLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanPartLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanPartLossLayer);
REGISTER_LAYER_CLASS(EuclideanPartLoss);

}  // namespace caffe
