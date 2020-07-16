#include "readFiles.hpp"
#include "Config.h"

#include "paddle_api.h"
#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <limits>

int WARMUP_COUNT = 0;
int REPEAT_COUNT = 1;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_HIGH;
const std::vector<int64_t> INPUT_SHAPE = {1, 3, 320, 320};
const std::vector<float> INPUT_MEAN = {0.5f, 0.5f, 0.5f};
const std::vector<float> INPUT_STD = {0.5f, 0.5f, 0.5f};
const float SCORE_THRESHOLD = 0.5f;

inline int64_t get_current_us() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

namespace tfdetector {

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

class TfdetImpl : public TFDetectorInterface{
public:
    TfdetImpl() = default;

    ~TfdetImpl() = default;

    TfdetImpl(const std::string& configFile);

    TfdetImpl(std::string modelname, int verbose, int number_of_threads, int saveImg, float threshold, std::string pathtoboxpriors);

    bool init();

    void run(cv::InputArray src);

    cv::Mat getboundingImage() const {
        return boundingImage_;
    }

    std::vector<object> getobjects() const {
        return instanceObject_;
    };

protected:

    cv::Mat boundingImage_;
    std::vector<object> instanceObject_;

    paddle::lite_api::MobileConfig config_;
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;

    cv::Mat frameImg_tf_;
    int originWidth_;
    int originHeight_;
    int originChannel_;
    std::string modelname_ = "";
    int verbose_ = 0;
    int number_of_threads_ = 4;
    int saveImg_ = 0;
    float threshold_1_ = 0.75;
    float threshold_2_ = 0.75;
    float threshold_3_ = 0.5;
    float threshold_4_ = 0.5;
    int accel_ = 1;
    std::string path_to_boxpriors_ = "";
    int classNum_;

    size_t predictionsNum = NUM_RESULTS * 4;
    size_t outputClassesNum;
    float boxPriors_[4][NUM_RESULTS];
    float *predictions_ = new float[predictionsNum];
    float *outputClasses_;
    int output_[2][NUM_RESULTS];
    float outputres_[NUM_RESULTS];

    void process(cv::Mat &input_image);
    void preprocess(cv::Mat &input_image, const std::vector<float> &input_mean,
                const std::vector<float> &input_std, int input_width,
                int input_height, float *input_data);
    float NMS_THRESHOLD_;

};

TfdetImpl::TfdetImpl(const std::string& configFile){
    cv::FileStorage fs(configFile, cv::FileStorage::READ);
    Config* config = Config::getInstance();
    fs.root() >> config;
    modelname_ = config->modelname;
    verbose_ = config->verbose;
    number_of_threads_ = config->number_of_threads;
    saveImg_ = config->saveImg;
    threshold_1_ = config->threshold1;
    threshold_2_ = config->threshold2;
    threshold_3_ = config->threshold3;
    threshold_4_ = config->threshold4;
    path_to_boxpriors_ = config->path_to_boxpriors;
    classNum_ = config->classNum;
    outputClassesNum = NUM_RESULTS * classNum_;
    outputClasses_ = new float[outputClassesNum];
    NMS_THRESHOLD_ = config->NMS;
    init();
}

TfdetImpl::TfdetImpl(std::string modelname, int verbose, int number_of_threads, int saveImg, float threshold, std::string pathtoboxpriors){
    modelname_ = modelname;
    verbose_ = verbose;
    number_of_threads_ = number_of_threads;
    saveImg_ = saveImg;
    threshold_1_ = threshold;
    threshold_2_ = threshold;
    threshold_3_ = threshold;
    threshold_4_ = threshold;
    path_to_boxpriors_ = pathtoboxpriors;
    classNum_ = 5;
    outputClassesNum = NUM_RESULTS * classNum_;
    outputClasses_ = new float[outputClassesNum];
    NMS_THRESHOLD_ = 0.45f;
    init();
}

bool TfdetImpl::init(){
    config_.set_model_from_file(modelname_);
    config_.set_threads(number_of_threads_);
    config_.set_power_mode(CPU_POWER_MODE);
    predictor_ = paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(config);
}

void TfdetImpl::run(cv::InputArray src){
    if(src.empty()){
            if(verbose_)
                std::cout << "input image is empty !!" << std::endl;
            return;
        }
    cv::Mat img;

    img = src.getMat();

    if(verbose_)
        std::cout << img.rows <<"         "<< img.cols << std::endl;
    
    cv::Mat processed_image = process(img);
}

void TfdetImpl::preprocess(cv::Mat &input_image, const std::vector<float> &input_mean,
                const std::vector<float> &input_std, int input_width,
                int input_height, float *input_data) {
  cv::Mat resize_image;
  cv::resize(input_image, resize_image, cv::Size(input_width, input_height), 0, 0);
  if (resize_image.channels() == 4) {
    cv::cvtColor(resize_image, resize_image, CV_BGRA2RGB);
  }
  cv::Mat norm_image;
  resize_image.convertTo(norm_image, CV_32FC3, 1 / 255.f);
  // NHWC->NCHW
  int image_size = input_height * input_width;
  const float *image_data = reinterpret_cast<const float *>(norm_image.data);
  float32x4_t vmean0 = vdupq_n_f32(input_mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(input_mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(input_mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(1.0f / input_std[0]);
  float32x4_t vscale1 = vdupq_n_f32(1.0f / input_std[1]);
  float32x4_t vscale2 = vdupq_n_f32(1.0f / input_std[2]);
  float *input_data_c0 = input_data;
  float *input_data_c1 = input_data + image_size;
  float *input_data_c2 = input_data + image_size * 2;
  int i = 0;
  for (; i < image_size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(image_data);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(input_data_c0, vs0);
    vst1q_f32(input_data_c1, vs1);
    vst1q_f32(input_data_c2, vs2);
    image_data += 12;
    input_data_c0 += 4;
    input_data_c1 += 4;
    input_data_c2 += 4;
  }
  for (; i < image_size; i++) {
    *(input_data_c0++) = (*(image_data++) - input_mean[0]) / input_std[0];
    *(input_data_c1++) = (*(image_data++) - input_mean[1]) / input_std[1];
    *(input_data_c2++) = (*(image_data++) - input_mean[2]) / input_std[2];
  }
}


void TfdetImpl::process(cv::Mat &input_image) {
    // Preprocess image and fill the data of input tensor
    std::unique_ptr<paddle::lite_api::Tensor> input_tensor(
        std::move(predictor->GetInput(0)));
    input_tensor->Resize(INPUT_SHAPE);
    int input_width = INPUT_SHAPE[3];
    int input_height = INPUT_SHAPE[2];
    auto *input_data = input_tensor->mutable_data<float>();
    double preprocess_start_time = get_current_us();
    preprocess(input_image, INPUT_MEAN, INPUT_STD, input_width, input_height,
                input_data);
    double preprocess_end_time = get_current_us();
    double preprocess_time = (preprocess_end_time - preprocess_start_time) / 1000.0f;

    std::unique_ptr<paddle::lite_api::Tensor> input_tensor1(
        std::move(predictor->GetInput(1))
    );
    input_tensor1->Resize({1,2});
    auto *data1 = input_tensor1->mutable_data<int>();
    data1[0] = input_image.rows;
    data1[1] = input_image.cols;

    double prediction_time;
    // Run predictor
    // warm up to skip the first inference and get more stable time, remove it in
    // actual products
    for (int i = 0; i < WARMUP_COUNT; i++) {
        predictor->Run();
    }
    // repeat to obtain the average time, set REPEAT_COUNT=1 in actual products
    double max_time_cost = 0.0f;
    double min_time_cost = std::numeric_limits<float>::max();
    double total_time_cost = 0.0f;
    for (int i = 0; i < REPEAT_COUNT; i++) {
        auto start = get_current_us();
        predictor->Run();
        auto end = get_current_us();
        double cur_time_cost = (end - start) / 1000.0f;
        if (cur_time_cost > max_time_cost) {
        max_time_cost = cur_time_cost;
        }
        if (cur_time_cost < min_time_cost) {
        min_time_cost = cur_time_cost;
        }
        total_time_cost += cur_time_cost;
        prediction_time = total_time_cost / REPEAT_COUNT;
        printf("iter %d cost: %f ms\n", i, cur_time_cost);
    }
    printf("warmup: %d repeat: %d, average: %f ms, max: %f ms, min: %f ms\n",
            WARMUP_COUNT, REPEAT_COUNT, prediction_time,
            max_time_cost, min_time_cost);

    // Get the data of output tensor and postprocess to output detected objects
    std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(
        std::move(predictor->GetOutput(0)));
    const float *output_data = output_tensor->mutable_data<float>();
    int64_t output_size = 1;
    for (auto dim : output_tensor->shape()) {
        output_size *= dim;
    }
    cv::Mat output_image = input_image.clone();
    double postprocess_start_time = get_current_us();
    std::vector<RESULT> results = postprocess(
        output_data, output_size, word_labels, SCORE_THRESHOLD, output_image, prediction_time);
    double postprocess_end_time = get_current_us();
    double postprocess_time = (postprocess_end_time - postprocess_start_time) / 1000.0f;

    printf("results: %d\n", results.size());
    for (int i = 0; i < results.size(); i++) {
        printf("[%d] %s - %f %f,%f,%f,%f\n", i, results[i].class_name.c_str(),
            results[i].score, results[i].left, results[i].top, results[i].right,
            results[i].bottom);
    }
    printf("Preprocess time: %f ms\n", preprocess_time);
    printf("Prediction time: %f ms\n", prediction_time);
    printf("Postprocess time: %f ms\n\n", postprocess_time);

    return output_image;
}


TFDetectorInterface* TFDetectorInterface::createDetector(const std::string& configFile){
    return new TfdetImpl(configFile);
}

TFDetectorInterface* TFDetectorInterface::createDetector(std::string modelname, int verbose, int number_of_threads, int saveImg, float threshold, std::string pathtopriors){
    return new TfdetImpl(modelname, verbose, number_of_threads, saveImg, threshold, pathtopriors);
}
}