package com.ifham.NeuralNetwork;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LoadAndSave {

  public static void saveModel(NeuralNetwork nn, String fileName) throws IOException {
    // 🔄 Sync weights & biases from neurons before saving
    for (Layer layer : nn.layers) {
      for (int i = 0; i < layer.neurons.length; i++) {
        layer.biases[i] = layer.neurons[i].bias;
        layer.weights[i] = layer.neurons[i].weights.clone(); // deep copy
      }
    }

    JSONObject model = new JSONObject();
    JSONArray layersArray = new JSONArray();

    for (Layer layer : nn.layers) {
      final JSONObject layerJson = getJsonObject(layer);
      layersArray.put(layerJson);
    }

    model.put("layers", layersArray);

    try (FileWriter writer = new FileWriter(fileName)) {
      writer.write(model.toString(4)); // Pretty print
      System.out.println("Model saved to JSON: " + fileName);
    }
  }

  private static JSONObject getJsonObject(Layer layer) {
    JSONObject layerJson = new JSONObject();

    JSONArray biasesArray = new JSONArray();
    for (double b : layer.biases) {
      biasesArray.put(b);
    }
    layerJson.put("biases", biasesArray);

    JSONArray weightsArray = new JSONArray();
    for (double[] row : layer.weights) {
      JSONArray rowArray = new JSONArray();
      for (double w : row) {
        rowArray.put(w);
      }
      weightsArray.put(rowArray);
    }
    layerJson.put("weights", weightsArray);

    return layerJson;
  }

  public static NeuralNetwork loadModel(String fileName) throws IOException {
    NeuralNetwork nn = new NeuralNetwork(); // no-arg constructor needed

    try (FileReader reader = new FileReader(fileName)) {
      char[] buffer = new char[8192];
      int length = reader.read(buffer);
      String jsonText = new String(buffer, 0, length);

      JSONObject model = new JSONObject(jsonText);
      JSONArray layersArray = model.getJSONArray("layers");

      List<Layer> loadedLayers = new ArrayList<>();

      for (int i = 0; i < layersArray.length(); i++) {
        JSONObject layerJson = layersArray.getJSONObject(i);

        JSONArray biasesArray = layerJson.getJSONArray("biases");
        double[] biases = new double[biasesArray.length()];
        for (int j = 0; j < biases.length; j++) {
          biases[j] = biasesArray.getDouble(j);
        }

        JSONArray weightsArray = layerJson.getJSONArray("weights");
        double[][] weights = new double[weightsArray.length()][];
        for (int j = 0; j < weightsArray.length(); j++) {
          JSONArray row = weightsArray.getJSONArray(j);
          weights[j] = new double[row.length()];
          for (int k = 0; k < row.length(); k++) {
            weights[j][k] = row.getDouble(k);
          }
        }

        Layer layer = new Layer(weights[0].length, biases.length); // inputSize, neuronCount
        layer.weights = weights;
        layer.biases = biases;
        layer.rebuildNeurons(); // 💡 Rebuild cim.ifham.NeuralNetwork.Neuron[] after weights are loaded
        loadedLayers.add(layer);
      }

      nn.layers = loadedLayers;
      System.out.println("Model loaded from JSON: " + fileName);
    }

    return nn;
  }
}
