#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>


int main([[maybe_unused]] int argc, [[maybe_unused]]  char **argv) {
    // Open video stream
	cv::VideoCapture video_capture{"/dev/video0"};
	cv::Mat captured_frame;
	cv::Mat blob;
	cv::Mat blob_reordered;

	const auto backends = cv::dnn::getAvailableBackends();
	for(auto& [backend, target] : backends) {
		std::cout << backend << " | " << target << "\n";
	}
	std::cout << "\n\n################\n";
	cv::dnn::Net net;
	try {
		net = cv::dnn::readNet("../blazepose/blazepose_full.onnx");
	}
	catch(const cv::Exception& e) {
		std::cerr << "1 failed. " << e.what();
	}
	std::cout << "\n\n################\n";
	std::vector<cv::Mat> out_blobs;
	std::vector<std::string> out_names{"Identity", "Identity_4"};

	while(true) {
	    bool status = video_capture.grab();
        if(status) {
            video_capture.retrieve(captured_frame);
			blob = cv::dnn::blobFromImage(captured_frame, 1 / 255.0, {256, 256}, {}, true, true, CV_32F);
			cv::transposeND(blob, {0, 2, 3, 1}, blob_reordered);
			net.setInput(blob_reordered);
			net.forward(out_blobs, out_names);
			std::cout << "################\n";
			std::cout << out_blobs[0].size << '\n';
			std::cout << out_blobs[1].size << '\n';
		}
		cv::imshow("Frame", captured_frame);
    	if(cv::waitKey(10) == 'q') {
			break;
		}
	}

    return 0;
}
