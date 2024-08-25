#include <vector>

using namespace std;

class Layer {
public:
	virtual ~Layer() = default;
	virtual void forward(const vector<float>& input, vector<float>& output) const = 0;
	virtual int getOutputDims() const = 0;
};

class FullyConnectedLayer : public Layer {
public:
	FullyConnectedLayer(int input_dims, int output_dims);
	~FullyConnectedLayer();
	void setWeights(const vector<vector<float>>& weights);
	void setBiases(const vector<float>& biases);
	void forward(const vector<float>& input, vector<float>& output) const override;
	int getOutputDims() const override;

private:
	int output_dims_;
	vector<vector<float>> weights_;
	vector<float> biases_;
};

class ReLU : public Layer {
public:
	ReLU(int input_dims);
	~ReLU();
	void forward(const vector<float>& input, vector<float>& output) const override;
	int getOutputDims() const override;

private:
	int output_dims_;
};

class Sigmoid : public Layer {
public:
	Sigmoid(int input_dims);
	~Sigmoid();
	void forward(const vector<float>& input, vector<float>& output) const override;
	int getOutputDims() const override;

private:
	int output_dims_;
};

class SoftMax : public Layer {
public:
	SoftMax(int input_dims);
	~SoftMax();
	void forward(const vector<float>& input, vector<float>& output) const override;
	int getOutputDims() const override;

private:
	int output_dims_;
};