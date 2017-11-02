#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SliceBatchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
}

template <typename Dtype>
void SliceBatchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	int width = bottom[0]->width();
	int height = bottom[0]->height();
	int channels = bottom[0]->channels();
	top[0]->Reshape(1, channels, height, width);
	top[1]->Reshape(1, channels, height, width);
}

template <typename Dtype>
void SliceBatchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data1 = top[0]->mutable_cpu_data();
    Dtype* top_data2 = top[1]->mutable_cpu_data();
    const Dtype* middle_data = bottom_data + bottom[0]->offset(1);
    caffe_copy(bottom[0]->count()/2, bottom_data, top_data1);
    caffe_copy(bottom[0]->count()/2, middle_data, top_data2);
}

template <typename Dtype>
void SliceBatchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom0_diff = bottom[0]->mutable_cpu_diff();
  caffe_copy(bottom[0]->count()/2, top_diff, bottom0_diff);
}

#ifdef CPU_ONLY
STUB_GPU(SliceBatchLayer);
#endif

INSTANTIATE_CLASS(SliceBatchLayer);
REGISTER_LAYER_CLASS(SliceBatch);

}  // namespace caffe
