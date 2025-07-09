package com.ifham.NeuralNetwork;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

class   Main{
  public static void main(String[] args) throws IOException {

  NeuralNetwork net ;
    double[][] inputs = {
            {0, 0, 1,1,0},
            {1, 1, 1,0,1},
            {1, 0, 1,1,0},
            {1, 1, 1,0,1}
    };

    double[][] targets = {
            {0, 1,0,0},
            {0, 1,1,0},
            {1, 0,1,1},
            {0, 1,1,0}
    };
String fileName="model.json";
    File file = new File(fileName);
    if(file.exists()){
      net = LoadAndSave.loadModel(fileName);
    }
    else {
      int[] layer = {5,4,4};
      net = new NeuralNetwork(layer);
      net.train(inputs, targets, 0.1, 60000);

      LoadAndSave.saveModel(net, fileName);
    }

    for (double[] input : inputs) {
      double[] out = net.predict(input);


      System.out.println("Input: " + Arrays.toString(input) + " → Output: " + Arrays.toString(out));
    }
    NeuralNetwork.THREAD_POOL.shutdown();


  }
}