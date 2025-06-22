public class Neuron {
    double[] weights;
    double bias;
    double output;
    double[] inputs;
    double learningRate;

    public Neuron(int inputSize, double learningRate) {
        this.weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = Math.random() * 2 - 1;
        }
        this.bias = Math.random() * 2 - 1;
        this.learningRate = learningRate;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }

    public double feedForward(double[] inputs) {
        this.inputs = inputs;
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += inputs[i] * weights[i];
        }
        output = sigmoid(sum);
        return output;
    }

    public void updateWeights(double error) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * error * inputs[i];
        }
        bias += learningRate * error;
    }
}
