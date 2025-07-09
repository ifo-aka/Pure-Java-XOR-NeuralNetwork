package com.ifham.NeuralNetwork;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;


public class Layer implements Serializable {
  @Serial
 private  static  final long serialVersionUID= 1L;
  double[][] weights;
  double[] biases;
  public  Neuron[] neurons;

  public Layer(int inputNeuron, int inputSize){
    neurons = new Neuron[inputNeuron];
    for (int i = 0; i < inputNeuron; i++) {
      neurons[i]= new Neuron(inputSize);
    }
    this.weights = new double[inputNeuron][inputSize];
    this.biases = new double[inputNeuron];
  }


  public double[] feedForward(double[] inputs) {
    double[] output = new double[neurons.length];
   if(inputs.length<7){
     for (int i = 0; i < neurons.length; i++) {
       output[i]=neurons[i].feedForward(inputs);
     }
   }else {
     List<Future<Void>> futures = new ArrayList<>();

     for (int i = 0; i < neurons.length; i++) {
       final int index = i;
       futures.add(NeuralNetwork.THREAD_POOL.submit(() -> {
         output[index] = neurons[index].feedForward(inputs);
         return null;
       }));
     }

     for (Future<Void> future : futures) {
       try {
         future.get();
       } catch (InterruptedException | ExecutionException e) {
         e.printStackTrace();
       }
     }
   }

    return output;
  }

  public double[] getOutputs() {
    double[] out = new double[neurons.length];
    for (int i = 0; i < neurons.length; i++) {
      out[i] = neurons[i].output;
    }
    return out;
  }
  public void rebuildNeurons() {
    neurons = new Neuron[biases.length];
    for (int i = 0; i < biases.length; i++) {
      neurons[i] = new Neuron(weights[i].length);
      neurons[i].weights = weights[i].clone(); // deep copy
      neurons[i].bias = biases[i];
    }
  }










}
