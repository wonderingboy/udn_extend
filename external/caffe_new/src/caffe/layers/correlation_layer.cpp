#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void CorrelationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int width = bottom[0]->width();
  int height = bottom[0]->height();

  top[0]->Reshape(num, 1, bottom[1]->height(), bottom[1]->width());
  sub_region.Reshape(1, channels, width, height);
  //LOG(INFO)<<width<<","<<height<<","<<channels;
  
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    
  //Reshape top
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int width = bottom[0]->width();
  int height = bottom[0]->height();

  top[0]->Reshape(num, 1, bottom[1]->height(), bottom[1]->width());
  sub_region.Reshape(1, channels, height, width);
  //LOG(INFO)<<width<<","<<height<<","<<channels;
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(CorrelationLayer);
#endif

INSTANTIATE_CLASS(CorrelationLayer);
REGISTER_LAYER_CLASS(Correlation);

}  // namespace caffe
