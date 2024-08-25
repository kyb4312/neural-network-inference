#include "NeuralNetwork.h"

TrainedNN::TrainedNN() = default;
TrainedNN::~TrainedNN() {
    for (auto layer : layers_) {
        delete layer;
    }
}

void TrainedNN::addLayer(Layer* layer) {
    layers_.push_back(layer);
}

void TrainedNN::predict(const vector<float>& input, vector<float>& output) const {
    vector<float> current_input = input;
    for (auto& layer : layers_) {
        vector<float> current_output(layer->getOutputDims());
        layer->forward(current_input, current_output);
        current_input = current_output;
    }
    output = current_input;
}