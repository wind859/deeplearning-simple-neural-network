#include "neural_network.h"

#include <math.h>
#include <random>
#include <ctime>

Neuron::Neuron(double bias) : m_bias(bias), m_output(0)
{
}

double Neuron::calculateOutput(const std::vector<double> & vecInputData)
{
    m_vecInputData = vecInputData;
    m_output = sigmoid(calculateTotalNetInput());
    return m_output;
}

double Neuron::calculateTotalNetInput()
{
    double dTotalValue = 0;
    for(unsigned int idx=0; idx<m_vecInputData.size(); idx++ )
    {
        dTotalValue += m_vecInputData[idx] * m_vecWeightData[idx];
    }
    return dTotalValue + m_bias;
}

double Neuron::sigmoid(double val)
{
    return 1 / (1 + exp(-val));
}

// 计算神经元误差，由最终输出与期望值的平方差决定
double Neuron::calculateError(double targetOutput)
{
    return 0.5 * (targetOutput - m_output) * (targetOutput - m_output);
}

// 计算偏导，网络输入对权重
double Neuron::calculatePDTotalNetInput2Weight(int inputIndex)
{
    if( inputIndex<0 || inputIndex>=(int)m_vecInputData.size() )
    {
        return 0;
    }
    return m_vecInputData[inputIndex];
}

// 计算偏导，网络输入对输入
double Neuron::calculatePDTotalNetInput2Input()
{
    return m_output * (1 - m_output);
}

// 计算偏导，误差对输出
double Neuron::calculatePDError2Output(double targetOutput)
{
    return -(targetOutput - m_output);
}

// 计算偏导，误差对网络输入
double Neuron::calculatePDError2TotalNetInput(double targetOutput)
{
    return calculatePDError2Output(targetOutput) * calculatePDTotalNetInput2Input();
}

void Neuron::appendWeight(double val)
{
    m_vecWeightData.push_back(val);
}

void Neuron::setWeight(double val, int idx)
{
    if( idx<0 || idx>=m_vecWeightData.size() )
    {
        return;
    }
    m_vecWeightData[idx] = val;
}

// 权重数
int Neuron::weightNumber()
{
    return (int)(m_vecWeightData.size());
}

// 权重值
double Neuron::weight(int idx)
{
    if( idx<0 || idx>=(int)(m_vecWeightData.size()))
        return 0;
    return m_vecWeightData[idx];
}

// 输出值
double Neuron::output()
{
    return m_output;
}

////////////////////////////////////////////////////////////////////////////////////
// NeuronLayer

NeuronLayer::NeuronLayer(int neuronNum, double bias)
{
    if( 0==bias )
    {
        m_bias = (double)(rand() % 100 ) / 100;
    }
    else
    {
        m_bias = bias;
    }

    for(int idx=0; idx<neuronNum; idx++ )
	{
        m_vecNeurons.push_back(new Neuron(bias));
	}
}

NeuronLayer::~NeuronLayer()
{
    for(int idx=0; idx<(int)(m_vecNeurons.size()); idx++ )
    {
        delete m_vecNeurons[idx];
    }
    m_vecNeurons.clear();
}

// 报告状态
void NeuronLayer::inspect()
{
    printf("neuron number: %d\n", (int)m_vecNeurons.size());
    for( int idx=0; idx<(int)(m_vecNeurons.size()); idx++ )
    {
        printf("    index = %d \n", idx);
        printf("        weight: \n");
        for( int i=0; i<m_vecNeurons[idx]->weightNumber(); i++)
        {
            printf("            idx = %d: value =%.3f \n", i, m_vecNeurons[idx]->weight(i));
        }
        printf("        bias: %.3f\n", m_bias);
    }
}

// 前向传播
std::vector<double> NeuronLayer::forward(const std::vector<double> & vecInputData)
{
    std::vector<double> vecOutputData;
    for( int idx=0; idx<(int)(m_vecNeurons.size()); idx++ )
    {
        vecOutputData.push_back(m_vecNeurons[idx]->calculateOutput(vecInputData));
    }
    return vecOutputData;
}

// 获取输出
std::vector<double> NeuronLayer::output()
{
    std::vector<double> vecOutputData;
    for( int idx=0; idx<(int)(m_vecNeurons.size()); idx++ )
    {
        vecOutputData.push_back(m_vecNeurons[idx]->output());
    }
    return vecOutputData;
}

// 神经元数目
int NeuronLayer::neuronNumer()
{
    return (int)m_vecNeurons.size();
}

// 获取神经元
Neuron * NeuronLayer::neuron(int idx)
{
    if( idx<0 || idx>=(int)(m_vecNeurons.size()) )
    {
        return nullptr;
    }
    return m_vecNeurons[idx];
}

////////////////////////////////////////////////////////////////////////////////////
// NeuralNetwork

const double NeuralNetwork::s_learnRadio = 0.5;

NeuralNetwork::NeuralNetwork(int inputNumber, int hiddenNumber, int outputNubmer, 
        const std::vector<double> & vecHiddenLayerWeight, double hiddenLayerBias,
        const std::vector<double> & vecOutputLayerWeight, double outputLayerBias)
{
    m_inputNumber = inputNumber;

    m_hiddenLayer = new NeuronLayer(hiddenNumber, hiddenLayerBias);
    m_outputLayer = new NeuronLayer(outputNubmer, outputLayerBias);

    initWeightsFromInputs2HiddenLayerNeurons(vecHiddenLayerWeight);
    initWeightsFromHiddenLayerNeurons2OutputLayerNeurons(vecOutputLayerWeight);
}

void NeuralNetwork::initWeightsFromInputs2HiddenLayerNeurons(const std::vector<double> & vecHiddenLayerWeight)
{
    int weightIndex = 0;
    for( int i=0; i<m_hiddenLayer->neuronNumer(); i++ )
    {
        Neuron * pNeuron = m_hiddenLayer->neuron(i);
        for(int j=0; j<m_inputNumber; j++ )
        {
            if( vecHiddenLayerWeight.empty() )
            {
                pNeuron->appendWeight((double)(rand() % 100 ) / 100);
            }
            else
            {
                pNeuron->appendWeight(vecHiddenLayerWeight[weightIndex]);
            }
            weightIndex++;
        }
    }
}

void NeuralNetwork::initWeightsFromHiddenLayerNeurons2OutputLayerNeurons(const std::vector<double> & vecOutputLayerWeight)
{
    int weightIndex = 0;
    for( int i=0; i<m_outputLayer->neuronNumer(); i++ )
    {
        Neuron * pNeuron = m_outputLayer->neuron(i);
        for(int j=0; j<m_hiddenLayer->neuronNumer(); j++ )
        {
            if( vecOutputLayerWeight.empty() )
            {
                pNeuron->appendWeight((double)(rand() % 100 ) / 100);
            }
            else
            {
                pNeuron->appendWeight(vecOutputLayerWeight[weightIndex]);
            }
            weightIndex++;
        }
    }
}

void NeuralNetwork::inspect()
{
    printf("neural network inspect:\n");
    printf("    input number: %d\n", m_inputNumber);
    printf("---- hidden layer ----\n");
    m_hiddenLayer->inspect();
    printf("---- output layer ----\n");
    m_outputLayer->inspect();
    printf("----------------------\n");
}

std::vector<double> NeuralNetwork::forward(const std::vector<double> & vecInputData)
{
    std::vector<double> vecHiddenLayerOutput = m_hiddenLayer->forward(vecInputData);
    return m_outputLayer->forward(vecHiddenLayerOutput);
}

void NeuralNetwork::train(const std::vector<double> & vecInputData, const std::vector<double> & vecGroundTruth)
{
    forward(vecInputData);

    // 1. 输出神经元的值
    std::vector<double> vecPDErrorOutput2TotalNetInput;
    for( int idx=0; idx<m_outputLayer->neuronNumer(); idx++ )
    {
        Neuron * pNeuron = m_outputLayer->neuron(idx);
        double dValue = pNeuron->calculatePDError2TotalNetInput(vecGroundTruth[idx]);
        vecPDErrorOutput2TotalNetInput.push_back(dValue);
    }

    // 2. 隐含层神经元的值
    std::vector<double> vecPDErrorHidden2TotalNetInput;
    for( int idx=0; idx<m_hiddenLayer->neuronNumer(); idx++ )
    {
        Neuron * pNeuron = m_hiddenLayer->neuron(idx);
        double dErrorHiddenNeuronOutput = 0;
        for( int i=0; i<m_outputLayer->neuronNumer(); i++ )
        {
            dErrorHiddenNeuronOutput += vecPDErrorOutput2TotalNetInput[i] * pNeuron->weight(idx);
        }
        double dValue = dErrorHiddenNeuronOutput * pNeuron->calculatePDTotalNetInput2Input();
        vecPDErrorHidden2TotalNetInput.push_back(dValue);
    }

    // 3. 更新输出层权重系数
    for( int i=0; i<m_outputLayer->neuronNumer(); i++ )
    {
        Neuron * pNeuron = m_outputLayer->neuron(i);
        for( int j=0; j<pNeuron->weightNumber(); j++ )
        {
            double dw = pNeuron->weight(j);
            double dPDError2Weight = vecPDErrorOutput2TotalNetInput[i] * pNeuron->calculatePDTotalNetInput2Weight(j);
            double dnw = dw - dPDError2Weight * s_learnRadio;
            pNeuron->setWeight(dnw, j);
        }
    }

    // 4. 更新隐含层的权重系数
    for( int i=0; i<m_hiddenLayer->neuronNumer(); i++ )
    {
        Neuron * pNeuron = m_hiddenLayer->neuron(i);
        for( int j=0; j<pNeuron->weightNumber(); j++ )
        {
            double dw = pNeuron->weight(j);
            double dPDError2Weight = vecPDErrorHidden2TotalNetInput[i] * pNeuron->calculatePDTotalNetInput2Weight(j);
            double dnw = dw - dPDError2Weight * s_learnRadio;
            pNeuron->setWeight(dnw, j);
        }
    }
}

double NeuralNetwork::calculateTotalError(const std::vector<double> & vecInputData, const std::vector<double> & vecGroundTruth)
{
    double dTotalError = 0;
    forward(vecInputData);
    for(int idx=0; idx<(int)(vecGroundTruth.size()); idx++ )
    {
        Neuron * p = m_outputLayer->neuron(idx);
        dTotalError += p->calculateError(vecGroundTruth[idx]);
    }
    return dTotalError;
}
