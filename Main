import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        XORNetwork network = new XORNetwork(0.5);

        double[][] inputs = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };

        double[] targets = {0, 1, 1, 0};

        // Training
        for (int epoch = 0; epoch < 100_000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                network.train(inputs[i], targets[i]);
            }
        }

        // Testing
        System.out.println("Testing XOR:");
        for (int i = 0; i < inputs.length; i++) {
            double result = network.feedForward(inputs[i]);
            System.out.printf("Input: %s → Output: %.4f\n", Arrays.toString(inputs[i]), result);
        }
    }
}
