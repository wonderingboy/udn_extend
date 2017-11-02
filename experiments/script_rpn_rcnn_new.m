function [train_box train_box_rcnn train_box_rpn]= script_rpn_rcnn_new()

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% opts.nms_overlap_thres      = 0.7;
opts.after_nms_topN         = 300;
opts.use_gpu                = true;
opts.test_scales            = 800;
%% -------------------- INIT_MODEL --------------------
model_dir_rcnn  = fullfile(pwd, 'def_model/final_nohnm.caffemodel'); 
rcnn_model_net = fullfile(pwd, 'def_model/test_fastrcnn.prototxt');


proposal_detection_model    = load_proposal_detection_model();
proposal_detection_model.detection_net_def ...
                                = rcnn_model_net;
proposal_detection_model.detection_net ...
                                = model_dir_rcnn;

proposal_detection_model.conf_proposal.test_scales = opts.test_scales;
proposal_detection_model.conf_detection.test_scales = opts.test_scales;
if opts.use_gpu
    proposal_detection_model.conf_detection.image_means = gpuArray(proposal_detection_model.conf_detection.image_means);
end

caffe.init_log(fullfile(pwd, 'caffe_log'));
% proposal net
fast_rcnn_net = caffe.Net(proposal_detection_model.detection_net_def, 'test');
fast_rcnn_net.copy_from(proposal_detection_model.detection_net);

% set gpu/cpu
if opts.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end       

%% -------------------- WARM UP --------------------
load('imdb/cache/imdb_caltech_test.mat')
rpn = load('rpn_0_2.mat');

train_box ={};
train_box_rcnn = {};
running_time = [];
for j = 1:length(rpn.dt_boxes)
    
    img_name = fullfile(imdb.image_dir, [imdb.image_ids{j} '.jpg']);
    im = imread(img_name);
    fprintf('%d: %s\n',j, img_name);
    if opts.use_gpu
        im = gpuArray(im);
    end
     aboxes = rpn.dt_boxes{j};
     aboxes(:,3) = aboxes(:,1)+aboxes(:,3);
     aboxes(:,4) = aboxes(:,2)+aboxes(:,4); 
    % test detection
    th = tic();
    if 0%proposal_detection_model.is_share_feature
        [boxes, scores]             = fast_rcnn_conv_feat_detect(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            rpn_net.blobs(proposal_detection_model.last_shared_output_blob_name), ...
            aboxes(:, 1:4), opts.after_nms_topN);
    else
        [boxes, scores]             = fast_rcnn_im_detect_our(proposal_detection_model.conf_detection, fast_rcnn_net, im, ...
            aboxes(:, 1:4), 5);
    end
    t_detection = toc(th);
    % visualize
    classes = proposal_detection_model.classes;
    boxes_cell = cell(length(classes), 1);
    thres = 0;
    for i = 1:length(boxes_cell)
        if(isempty(boxes))
            continue;
        end
         try
        boxes_cell{i} = [boxes(:, (1+(i-1)*4):(i*4)), scores(:, i)];
        train_box_rcnn{j} = boxes_cell{i};
         catch
             pause
         end   
         boxes_cell{i} = boxes_cell{i}(nms(boxes_cell{i}, 0.3), :);
         I = boxes_cell{i}(:, 5) >= thres;
        boxes_cell{i} = boxes_cell{i}(I, :);
    end
    train_box{j} = boxes_cell;
end
save('test_rst', 'train_box_rcnn', 'train_box');
caffe.reset_all(); 
% clear mex;
end

function proposal_detection_model = load_proposal_detection_model()
    ld                          = load(fullfile(pwd, 'final/model_setting.mat'));
    proposal_detection_model    = ld.proposal_detection_model;
    clear ld;    
end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);       
    end
    if after_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
    end
end
