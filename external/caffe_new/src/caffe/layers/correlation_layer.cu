#include <vector>
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"
using std::max;
using std::min;
using std::floor;

namespace caffe {

// Copy (one line per thread) from one array to another, with arbitrary
// strides in the last two dimensions.
template <typename Dtype>
__global__ void copy_kernel(const int nthreads, const int x_center, const int y_center,
    const int candidate_width, const int candidate_height, const int search_width, const int search_height, const int channels, const int halfwidth, const int halfheight,   
    const Dtype* src, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % candidate_width;
    int h = (index / candidate_width)%candidate_height;
    int c = (index / candidate_width / candidate_height) % channels;
    int n = index / candidate_width / candidate_height / channels;

    int new_w_c = x_center + w - halfwidth;
    int new_h_c = y_center + h - halfheight;
	if (new_w_c < 0 || new_w_c >= search_width || new_h_c < 0 || new_h_c >= search_height){
		dest[index] = 0; 
	}else{        
		dest[index] = src[n * channels*search_height*search_width + c * search_height * search_width + new_h_c * search_width + new_w_c];
	}	    
  }
}

// Back the a to b
// src is a and the the dest b
template <typename Dtype>
__global__ void back_kernel(const int nthreads, const int x_center, const int y_center,
    const int candidate_width, const int candidate_height, const int search_width, const int search_height, const int channels, const int halfwidth, const int halfheight,
    const Dtype* src, Dtype* dest) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = index % candidate_width;
    int h = (index / candidate_width)%candidate_height;
    int c = (index / candidate_width / candidate_height) % channels;
    int n = index / candidate_width / candidate_height / channels;

    int new_w_c = x_center + w - halfwidth;
    int new_h_c = y_center + h - halfheight;

    int dest_index = n * channels*search_height*search_width + c * search_height * search_width + new_h_c * search_width + new_w_c;
    if (new_w_c < 0 || new_w_c >= search_width || new_h_c < 0 || new_h_c >= search_height){
         dest[dest_index] = dest[dest_index];
    }else{
         dest[dest_index] = dest[dest_index] + src[index];
    }
  }
}
 
template <typename Dtype>
void CorrelationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_candidate = bottom[0]->gpu_data(); 
  const Dtype* bottom_search = bottom[1]->gpu_data();
  Dtype* sub_region_data = sub_region.mutable_gpu_data(); 
  //const Dtype* sub_region_val = sub_region.cpu_data();


  int candidate_width = bottom[0]->width();
  int candidate_height = bottom[0]->height();  
  int channels = bottom[0]->channels();
  int num = bottom[0]->num();
  int lines = bottom[0]->count()/num;

  int search_width = bottom[1]->width();
  int search_height = bottom[1]->height();
  int halfwidth = floor(candidate_width/2);  
  int halfheight = floor(candidate_height/2);

//  top[0]->Reshape(num, 1, search_width, search_height);
//  while(1);
  //LOG(INFO)<<top[0]->num()<<top[0]->channels();
  //
  Dtype* top_data = top[0]->mutable_cpu_data();
  for(int i=0; i<num; i++){
      Dtype* top_data_index = top_data + top[0]->offset(i); 
      const Dtype* bottom_candidate_index = bottom_candidate + bottom[0]->offset(i);
      const Dtype* bottom_search_index = bottom_search + bottom[1]->offset(i);
      
      Dtype bottom_candidate_norm = 0;
      caffe_gpu_dot(lines, bottom_candidate_index, bottom_candidate_index, &bottom_candidate_norm);
     // LOG(INFO)<< bottom_search_index;
      for(int x_start = 0; x_start < search_width; ++x_start){
	for(int y_start = 0; y_start < search_height; ++y_start){
            Dtype rst = 0;
            //LOG(INFO)<<i<<",x: "<<x_start<<",y: "<<y_start;
	    copy_kernel<Dtype><<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(lines, x_start, y_start, candidate_width, candidate_height, search_width, search_height, channels, halfwidth, halfheight, bottom_search_index, sub_region_data);
            caffe_gpu_dot(lines, sub_region_data, bottom_candidate_index, &rst);

            Dtype bottom_search_norm = 0;
            caffe_gpu_dot(lines, sub_region_data, sub_region_data, &bottom_search_norm);

/*	    if (x_start == 10 && y_start == 10){ 
              Dtype* sub_region_val = sub_region.mutable_cpu_data();
              for(int layer_count = 2; layer_count <3; layer_count++){ 
               	for(int sub_i=0; sub_i<7; ++sub_i){
		    for(int sub_j=0; sub_j<7; ++sub_j){
		        LOG(INFO)<<sub_i<<","<<sub_j<<": "<<sub_region_val[layer_count*49 + sub_i*7+sub_j];
	            }
                }
              }
              LOG(INFO)<<y_start*search_width + x_start<<": "<<rst;
            }*/
            //LOG(INFO)<<y_start*search_width + x_start<<": "<<rst;
            top_data_index[y_start*search_width + x_start] = rst / sqrt(bottom_candidate_norm + 1) / sqrt(bottom_search_norm + 1);
	}
      }
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_candidate = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff_big = bottom[1]->mutable_gpu_diff();
  Dtype* sub_region_data = sub_region.mutable_gpu_data(); 
  const Dtype* bottom_search = bottom[1]->gpu_data();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int candidate_width = bottom[0]->width();
  int candidate_height = bottom[0]->height();  
  int search_width = bottom[1]->width();
  int search_height = bottom[1]->height();
  
  int lines = bottom[0]->count()/num;
  int halfwidth = floor(candidate_width/2);
  int halfheight = floor(candidate_height/2);

//  LOG(INFO)<<"top_diff size:"<<top[0]->num()<<top[0]->channels();
  if (propagate_down[0]) {
       for(int i=0; i<num; ++i){
          const Dtype* top_diff_index = top_diff + top[0]->offset(i);  
          Dtype* bottom_candidate_diff_index = bottom_diff + bottom[0]->offset(i);
          Dtype* bottom_search_diff_index = bottom_diff_big + bottom[1]->offset(i);
          const Dtype* bottom_search_index = bottom_search + bottom[1]->offset(i);
                   
 
          const Dtype* bottom_candidate_index = bottom_candidate + bottom[0]->offset(i);
          Dtype bottom_candidate_norm = 0;
          caffe_gpu_dot(lines, bottom_candidate_index, bottom_candidate_index, &bottom_candidate_norm);

	  for(int x_start = 0; x_start < search_width; ++x_start){
		for(int y_start = 0; y_start < search_height; ++y_start){
			copy_kernel<Dtype><<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(lines, x_start, y_start, candidate_width, candidate_height, search_width, search_height, channels,halfwidth,halfheight, bottom_search_index, sub_region_data);

                        Dtype bottom_search_norm = 0;
                        caffe_gpu_dot(lines, sub_region_data, sub_region_data, &bottom_search_norm);

			caffe_gpu_axpy(
			    lines,              // count
			    top_diff_index[y_start*search_width + x_start] / sqrt(bottom_candidate_norm + 1) / sqrt(bottom_search_norm + 1),                              // alpha
			    sub_region_data,                   // a			 
			    bottom_candidate_diff_index);  // b
                       // if(x_start == 10 && y_start == 10){
		       //     LOG(INFO)<<"diff: "<<y_start<<","<<x_start<<","<<top_diff_index[y_start*search_width+x_start];
			//}
                        // for the search diff
                       /*caffe_gpu_set(lines, Dtype(0), sub_region_data);
                       caffe_gpu_axpy(
                            lines,              // count
                            top_diff_index[y_start*search_width + x_start],                              // alpha
                            bottom[0]->gpu_data(),                   // a                      
                            sub_region_data);  
                       back_kernel<Dtype><<<CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS>>>(lines, x_start, y_start, candidate_width, candidate_height, search_width, search_height, channels,halfwidth,halfheight, sub_region_data, bottom_search_diff_index);*/
		}
	  }
       }
  CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CorrelationLayer);

}  // namespace caffe
