
#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#include <vector>

// 神经元
class Neuron
{
public:
    Neuron(double bias);

    // 计算神经元最终产生的输出信号
    double calculateOutput(const std::vector<double> & vecInputData);

    // 计算神经元误差，由最终输出与期望值的平方差决定
    double calculateError(double targetOutput);

    // 计算偏导，网络输入对权重
    double calculatePDTotalNetInput2Weight(int inputIndex);

    // 计算偏导，网络输入对输入
    double calculatePDTotalNetInput2Input();

    // 计算偏导，误差对输出
    double calculatePDError2Output(double targetOutput);

    // 计算偏导，误差对网络输入
    double calculatePDError2TotalNetInput(double targetOutput);

    // 添加权重
    void appendWeight(double val);

    // 设置权重
    void setWeight(double val, int idx);
    
    // 权重数
    int weightNumber();

    // 权重值
    double weight(int idx);

    // 输出值
    double output();

private:
    // 计算网络输入产生的信号值
    double calculateTotalNetInput();
    // 激活函数sigmoid
    double sigmoid(double val);

private:
    // 偏置
    double m_bias;
    // 权重
    std::vector<double> m_vecWeightData;
    // 输入
    std::vector<double> m_vecInputData;
    // 神经元最终输出
    double m_output;
};

class NeuronLayer
{
public:
    NeuronLayer(int neuronNum, double bias = 0);
    virtual ~NeuronLayer();

    // 报告状态
    void inspect();

    // 前向传播
    std::vector<double> forward(const std::vector<double> & vecInputData);

    // 获取输出
    std::vector<double> output();

    // 神经元数目
    int neuronNumer();

    // 获取神经元
    Neuron * neuron(int idx);

private:
    // 偏置
    double m_bias;

    // 神经元
    std::vector<Neuron*> m_vecNeurons;
};

class NeuralNetwork
{
public:
    static const double s_learnRadio;

    NeuralNetwork(int inputNumber, int hiddenNumber, int outputNubmer, 
        const std::vector<double> & vecHiddenLayerWeight, double hiddenLayerBias, 
        const std::vector<double> & vecOutputLayerWeight, double outputLayerBias);

    // 报告状态
    void inspect();

    // 网络训练
    void train(const std::vector<double> & vecInputData, const std::vector<double> & vecGroundTruth);

    // 计算损失
    double calculateTotalError(const std::vector<double> & vecInputData, const std::vector<double> & vecGroundTruth);

private:
    void initWeightsFromInputs2HiddenLayerNeurons(const std::vector<double> & vecHiddenLayerWeight);
    void initWeightsFromHiddenLayerNeurons2OutputLayerNeurons(const std::vector<double> & vecOutputLayerWeight);

    std::vector<double> forward(const std::vector<double> & vecInputData);

private:

    NeuronLayer * m_hiddenLayer;
    NeuronLayer * m_outputLayer;
    int m_inputNumber;
};

#endif