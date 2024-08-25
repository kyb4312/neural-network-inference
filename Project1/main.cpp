#include <iostream>
#include <vector>
#include <cstdlib>
#include "NeuralNetwork.h"

using namespace std;

TrainedNN getTrainedNN();

int main() {
    // 추론할 NN 불러오기
    TrainedNN trained_nn = getTrainedNN();
    NeuralNetwork* network = &trained_nn;

    // 추론할 raw_input 정의 (28 x 28 grayscale img)
    const int kRow = 28;
    const int kCol = 28;
    int raw_input[kRow][kCol];

    for (int i = 0; i < kRow; i++) {
        for (int j = 0; j < kCol; j++) {
            raw_input[i][j] = rand() % 256;
        }
    }

    // intput 가공
    vector<float> input(kRow * kCol);

    for (int i = 0; i < kRow; i++) {
        for (int j = 0; j < kCol; j++) {
            input[i * kCol + j] = raw_input[i][j];
        }
    }

    // inference
    const int kOutputSize = 10;
    vector<float> output(kOutputSize);

    network->predict(input, output);

    // 결과 출력
    cout.precision(2);
    for (int i = 0; i < kOutputSize; i++) {
        cout << i << ": " << output[i] << endl;
    }

    return 0;
}

TrainedNN getTrainedNN() {
    // 학습된 NN 모델 정의 (28 x 28 grayscale img 분류 모델)
    TrainedNN trained_nn;

    const int kTensorSize0 = 784;
    const int kTensorSize1 = 196;
    const int kTensorSize2 = 49;
    const int kTensorSize3 = 10; // 분류 label 개수

    // 첫 번째 fully connected layer 정의
    FullyConnectedLayer* fully_connected_1 = new FullyConnectedLayer(kTensorSize0, kTensorSize1);

    vector<vector<float>> weights_1(kTensorSize0, vector<float>(kTensorSize1, 0));
    for (int i = 0; i < kTensorSize0; i++) {
        for (int j = 0; j < kTensorSize1; j++) {
            weights_1[i][j] = (rand() % 10) / 5.0f - 1;
        }
    }
    fully_connected_1->setWeights(weights_1);

    vector<float> biases_1(kTensorSize1);
    for (int i = 0; i < kTensorSize1; i++) {
        biases_1[i] = (rand() % 10) / 5.0f - 1;
    }
    fully_connected_1->setBiases(biases_1);

    // 첫 번째 activation function 정의
    ReLU* activation_1 = new ReLU(kTensorSize1);

    // 두 번째 fully connected layer 정의
    FullyConnectedLayer* fully_connected_2 = new FullyConnectedLayer(kTensorSize1, kTensorSize2);

    vector<vector<float>> weights_2(kTensorSize1, vector<float>(kTensorSize2));
    for (int i = 0; i < kTensorSize1; i++) {
        for (int j = 0; j < kTensorSize2; j++) {
            weights_2[i][j] = (rand() % 10) / 5.0f - 1;
        }
    }
    fully_connected_2->setWeights(weights_2);

    vector<float> biases_2(kTensorSize2);
    for (int i = 0; i < kTensorSize2; i++) {
        biases_2[i] = (rand() % 10) / 5.0f - 1;
    }
    fully_connected_2->setBiases(biases_2);

    // 두 번째 activation function 정의
    ReLU* activation_2 = new ReLU(kTensorSize2);

    // 세 번째 fully connected layer 정의
    FullyConnectedLayer* fully_connected_3 = new FullyConnectedLayer(kTensorSize2, kTensorSize3);

    vector<vector<float>> weights_3(kTensorSize2, vector<float>(kTensorSize3));
    for (int i = 0; i < kTensorSize2; i++) {
        for (int j = 0; j < kTensorSize3; j++) {
            weights_3[i][j] = (rand() % 10) / 5.0f - 1;
        }
    }
    fully_connected_3->setWeights(weights_3);

    vector<float> biases_3(kTensorSize3);
    for (int i = 0; i < kTensorSize3; i++) {
        biases_3[i] = (rand() % 10) / 5.0f - 1;
    }
    fully_connected_3->setBiases(biases_3);

    // 세 번째 activation function 정의
    Sigmoid* activation_3 = new Sigmoid(kTensorSize3);

    // 정규화
    SoftMax* normalization = new SoftMax(kTensorSize3);

    trained_nn.addLayer(fully_connected_1);
    trained_nn.addLayer(activation_1);
    trained_nn.addLayer(fully_connected_2);
    trained_nn.addLayer(activation_2);
    trained_nn.addLayer(fully_connected_3);
    trained_nn.addLayer(activation_3);
    trained_nn.addLayer(normalization);

    return trained_nn;
}