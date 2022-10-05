#include "neural_network.h"

int main(int argc, char * argv[] )
{
    int inputNumber = 2;
    int hiddenNumber = 2;
    int outputNubmer = 2;
    
    std::vector<double> vecHiddenLayerWeight;
    vecHiddenLayerWeight.push_back(0.15);
    vecHiddenLayerWeight.push_back(0.2);
    vecHiddenLayerWeight.push_back(0.25);
    vecHiddenLayerWeight.push_back(0.3);
    
    double hiddenLayerBias = 0.35;
    std::vector<double> vecOutputLayerWeight;
    vecOutputLayerWeight.push_back(0.4);
    vecOutputLayerWeight.push_back(0.45);
    vecOutputLayerWeight.push_back(0.5);
    vecOutputLayerWeight.push_back(0.55);
    double outputLayerBias = 0.6;

    std::vector<double> vecInputData;
    vecInputData.push_back(0.05);
    vecInputData.push_back(0.1);

    std::vector<double> vecGroundTruth;
    vecGroundTruth.push_back(0.01);
    vecGroundTruth.push_back(0.09);

    NeuralNetwork network(inputNumber, hiddenNumber, outputNubmer,
        vecHiddenLayerWeight, hiddenLayerBias, 
        vecOutputLayerWeight, outputLayerBias);

    for( int idx=0; idx<10000; idx++ )
    {
        network.train(vecInputData, vecGroundTruth);
        printf("train cycle: %d, error = %.6f\n", idx,
            network.calculateTotalError(vecInputData, vecGroundTruth));
    }

    return 0;
}