#include "Layer.h"
#include <cmath>
#include <cstdlib>

// FullyConnectedLayer 클래스 구현
FullyConnectedLayer::FullyConnectedLayer(int input_dims, int output_dims) : output_dims_(output_dims) {
	weights_.resize(input_dims, vector<float>(output_dims, 0.0f));
	biases_.resize(output_dims, 0.0f);
}

FullyConnectedLayer::~FullyConnectedLayer() = default;

void FullyConnectedLayer::setWeights(const vector<vector<float>>& weights) {
	weights_ = weights;
}

void FullyConnectedLayer::setBiases(const vector<float>& biases) {
	biases_ = biases;
}

void FullyConnectedLayer::forward(const vector<float>& input, vector<float>& output) const {
	const int input_dims = input.size();
	const int output_dims = output.size();
	float sum = 0.0f;

	for (int i = 0; i < output_dims; i++) {
		sum = biases_[i];
		for (int j = 0; j < input_dims; j++) {
			sum += input[j] * weights_[j][i];
		}
		output[i] = sum;
	}
}

int FullyConnectedLayer::getOutputDims() const {
	return output_dims_;
}

// ReLU 클래스 구현
ReLU::ReLU(int input_dims) : output_dims_(input_dims) {}

ReLU::~ReLU() = default;

void ReLU::forward(const vector<float>& input, vector<float>& output) const {
	const int dims = input.size();

	for (int i = 0; i < dims; i++) {
		output[i] = (input[i] > 0) ? input[i] : 0;
	}
}

int ReLU::getOutputDims() const {
	return output_dims_;
}

// Sigmoid 클래스 구현
Sigmoid::Sigmoid(int input_dims) : output_dims_(input_dims) {}

Sigmoid::~Sigmoid() = default;

void Sigmoid::forward(const vector<float>& input, vector<float>& output) const {
	const int dims = input.size();

	for (int i = 0; i < dims; i++) {
		output[i] = 1.0 / (1.0 + exp(-input[i]));
	}
}

int Sigmoid::getOutputDims() const {
	return output_dims_;
}

//SoftMax 클래스 구현
SoftMax::SoftMax(int input_dims) : output_dims_(input_dims) {}

SoftMax::~SoftMax() = default;

void SoftMax::forward(const vector<float>& input, vector<float>& output) const {
	const int dims = input.size();
	float sum_exp = 0.0f;

	for (int i = 0; i < dims; i++) {
		sum_exp += exp(input[i]);
	}
	for (int i = 0; i < dims; i++) {
		output[i] = exp(input[i]) / sum_exp;
	}
}

int SoftMax::getOutputDims() const {
	return output_dims_;
}