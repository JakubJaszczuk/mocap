#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>


int main([[maybe_unused]] int argc, [[maybe_unused]]  char **argv) {
    // Open video stream
	cv::VideoCapture video_capture{"/dev/video0"};
	cv::Mat captured_frame;

	const auto backends = cv::dnn::getAvailableBackends();
	for(auto& [backend, target] : backends) {
		std::cout << backend << " | " << target << "\n";
	}
	std::cout << "################\n";
	try {
		auto net1 = cv::dnn::readNet("../movenet_singlepose_lightning_4/saved_model.pb");
	}
	catch(const cv::Exception& e) {
		std::cerr << "First failed. " << e.what();
	}
	std::cout << "################\n";
	try {
		auto net2 = cv::dnn::readNet("../movenet_singlepose_lightning_4/frozen_graph.pb");
	}
	catch(const cv::Exception& e) {
		std::cerr << "Second failed. " << e.what();
	}
	std::cout << "################\n";
	try {
		auto net3 = cv::dnn::readNet("../movenet_singlepose_lightning_4/model.onnx");
	}
	catch(const cv::Exception& e) {
		std::cerr << "Third failed. " << e.what();
	}
	std::cout << "################\n";

	while(true) {
	    bool status = video_capture.grab();
        if(status) {
            video_capture.retrieve(captured_frame);
		}
		cv::imshow("Frame", captured_frame);
    	if(cv::waitKey(10) == 'q') {
			break;
		}
	}

    return 0;
}
