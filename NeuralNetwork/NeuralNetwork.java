package com.ifham.NeuralNetwork;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class NeuralNetwork implements Serializable {
  @Serial
  private  static  final long serialVersionUID= 1L;
  public static final ExecutorService THREAD_POOL = Executors.newFixedThreadPool(
          Runtime.getRuntime().availableProcessors()
  );
  List<Layer> layers = new ArrayList<>();
  private  double[] lastInputs;

  public NeuralNetwork() {
    this.layers = new ArrayList<>();
  }

  public  NeuralNetwork(int[] inputSizes){
    for (int i = 0; i < inputSizes.length; i++) {
      int inputCount= i==0 ? inputSizes[i]: inputSizes[i-1];
      layers.add(new Layer(inputSizes[i],inputCount));

    }

  }
  public double[] predict(double[] inputs){
  this.lastInputs= inputs;
    double [] outputs= inputs;
    for(Layer layer : layers){
      outputs= layer.feedForward(outputs);
    }
  return  outputs;


  }



  public  void train(double[][] inputs ,double[][] targets,double learningRate,int epoch){

    for (int e = 0; e < epoch; e++) {
      double totalLoss= 0.0;
      for (int i = 0; i < inputs.length; i++) {
        double[] output= predict(inputs[i]);


        double[] target= targets[i];

        for (int j = 0; j < output.length; j++) {
          double error= target[j]- output[j];
          totalLoss+= error *error;

        }
        backPropagate(target,learningRate);


      }
      if(epoch %1000==0){
        System.out.println("Epoch"+ e+ " loss "+ totalLoss);
      }

    }

  }

  public void backPropagate(double[] target, double learningRate) {
    int last = layers.size() - 1;
    Layer outputLayer = layers.get(last);

    for (int i = 0; i < outputLayer.neurons.length; i++) {
      Neuron neuron = outputLayer.neurons[i];
      double output = neuron.output;
      double error = target[i] - output;
      neuron.delta = error * output * (1 - output);
    }

    for (int l = last - 1; l >= 0; l--) {
      Layer current = layers.get(l);
      Layer next = layers.get(l + 1);

      for (int i = 0; i < current.neurons.length; i++) {
        Neuron neuron = current.neurons[i];
        double output = neuron.output;

        double sum = 0.0;
        for (Neuron nextNeuron : next.neurons) {
          sum += nextNeuron.weights[i] * nextNeuron.delta;
        }

        neuron.delta = sum * output * (1 - output);
      }
    }

    for (int l = 0; l < layers.size(); l++) {
      Layer layer = layers.get(l);
      double[] inputs = (l == 0) ? lastInputs : layers.get(l - 1).getOutputs();

      for (Neuron neuron : layer.neurons) {
        neuron.updateWeights(inputs, learningRate);
      }
    }
  }









}
