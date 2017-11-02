#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MaxMaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  out_threshold_ = this->layer_param_.maxmask_param().out_threshold();
  //top_k_ = this->layer_param_.argmax_param().top_k();
  //CHECK_GE(top_k_, 1) << " top k must not be less than 1.";
  //CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
  //    << "top_k must be less than or equal to the number of classes.";
}

template <typename Dtype>
void MaxMaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //if (out_max_val_) {
    // Produces max_ind and max_val
  //  top[0]->Reshape(bottom[0]->num(), 2, top_k_, 1);
  //} else {
    // Produces only max_ind
    //top[0]->Reshape(bottom[0]->num(), 1, top_k_, 1);
  //}
   top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void MaxMaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int width = bottom[0]->width();
  int height = bottom[0]->height();
  int count = top[0]->count();
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < height; ++j) {
        for (int k = 0; k< width; ++k) {
	    Dtype max_val = 0;
            for (int l = 1; l < channels; ++l) {
		const Dtype* curr_val = bottom_data + bottom[0]->offset(i,l,j,k);
                if (*curr_val > max_val) { max_val = *curr_val;}
            }
            if (max_val < out_threshold_){
		Dtype* top_curr_des = top_data + top[0]->offset(i,0,j,k);
                const Dtype* curr_label = bottom_label + bottom[1]->offset(i,0,j,k);
	        *top_curr_des = *curr_label;
                //LOG(INFO)<<i<<" "<<j<<" "<<k;
	    }	    
        }      
    }
  }
//  caffe_mul(count, bottom_label, top[0]->cpu_data(), top[0]->mutable_cpu_data());
  //LOG(INFO)<<top[0]->count();
//  while(1);
}

INSTANTIATE_CLASS(MaxMaskLayer);
REGISTER_LAYER_CLASS(MaxMask);

}  // namespace caffe
