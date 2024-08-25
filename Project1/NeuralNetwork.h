#include <vector>
#include "Layer.h"

using namespace std;

class NeuralNetwork {
public:
    virtual ~NeuralNetwork() = default;
    virtual void addLayer(Layer* layer) = 0;
    virtual void predict(const vector<float>& input, vector<float>& output) const = 0;
protected:
    vector<Layer*> layers_;
};

class TrainedNN : public NeuralNetwork {
public:
    TrainedNN();
    ~TrainedNN();
    void addLayer(Layer* layer) override;
    void predict(const vector<float>& input, vector<float>& output) const override;
};