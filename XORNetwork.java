public class XORNetwork {
    Neuron[] hidden;
    Neuron outputNeuron;

    public XORNetwork(double learningRate) {
        hidden = new Neuron[2];
        hidden[0] = new Neuron(2, learningRate);
        hidden[1] = new Neuron(2, learningRate);
        outputNeuron = new Neuron(2, learningRate);
    }

    public double feedForward(double[] inputs) {
        double[] hiddenOutputs = new double[2];
        for (int i = 0; i < hidden.length; i++) {
            hiddenOutputs[i] = hidden[i].feedForward(inputs);
        }
        return outputNeuron.feedForward(hiddenOutputs);
    }

    public void train(double[] inputs, double target) {
        double[] hiddenOutputs = new double[2];
        for (int i = 0; i < hidden.length; i++) {
            hiddenOutputs[i] = hidden[i].feedForward(inputs);
        }

        double output = outputNeuron.feedForward(hiddenOutputs);
        double error = target - output;
        double outputGradient = error * output * (1 - output);

        // Update output neuron
        outputNeuron.updateWeights(outputGradient);

        // Backpropagate to hidden neurons
        for (int i = 0; i < hidden.length; i++) {
            double hiddenGradient = outputGradient * outputNeuron.weights[i] * hiddenOutputs[i] * (1 - hiddenOutputs[i]);
            hidden[i].updateWeights(hiddenGradient);
        }
    }
}
