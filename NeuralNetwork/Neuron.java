package com.ifham.NeuralNetwork;

import java.io.Serial;
import java.io.Serializable;

public class Neuron implements Serializable {
  @Serial
  private static  final  long serialVersionUID=1L;
// variable initialization
  double[] weights;
  double bias;
  double output;
  double delta;

  public Neuron(int inputSize) {
    weights= new double[inputSize];
    for (int i = 0; i < inputSize; i++) {
      weights[i]= Math.random()*2-1;


    }
    bias = Math.random()*2-1;

  }
//    Activation functions
  public  double activate(double x){
    return  1.0 / (1.0 +Math.exp(-x));
  }
  public  double activationDerivative(double x){
    double out = activate(x);
    return  out * (1- out);
  }
  public  double  feedForward(double[] inputs){
    double sum=0;
    for (int i = 0; i < inputs.length; i++) {
      sum+=weights[i] * inputs[i];


    }
    sum +=bias;
    output = activate(sum);
    return  output;

  }
  public void updateWeights(double[] inputs ,double learningRate){
    for (int i = 0; i < weights.length; i++) {
      weights[i]+= learningRate *  delta * inputs[i];
    }
    bias += learningRate * delta;
  }



}
