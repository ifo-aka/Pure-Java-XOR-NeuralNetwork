package com.ifham.NeuralNetwork;

import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Stage;

import java.io.File;
import java.util.Arrays;

public class MainGui extends Application {

  private NeuralNetwork net;

  @Override
  public void start(Stage primaryStage) {
    try {
      // Load trained model
      File file = new File("model.json");
      if (!file.exists()) {
        Alert alert = new Alert(Alert.AlertType.ERROR, "Trained model not found.");
        alert.showAndWait();
        return;
      }

      net = LoadAndSave.loadModel("model.json");

      // Create UI elements
      TextField[] inputFields = new TextField[5];
      for (int i = 0; i < inputFields.length; i++) {
        inputFields[i] = new TextField();
        inputFields[i].setPromptText("Input " + (i + 1));
      }

      Button predictBtn = new Button("Predict");
      Label outputLabel = new Label("Output will appear here...");

      predictBtn.setOnAction(e -> {
        try {
          double[] inputs = new double[5];
          for (int i = 0; i < 5; i++) {
            inputs[i] = Double.parseDouble(inputFields[i].getText());
          }

          double[] output = net.predict(inputs);
          outputLabel.setText("Output: " + Arrays.toString(output));
        } catch (NumberFormatException ex) {
          outputLabel.setText("Please enter valid numbers.");
        }
      });

      VBox inputBox = new VBox(10);
      inputBox.getChildren().addAll(inputFields);
      VBox root = new VBox(15, inputBox, predictBtn, outputLabel);
      root.setPadding(new Insets(20));

      Scene scene = new Scene(root, 400, 300);
      primaryStage.setTitle("Neural Network GUI");
      primaryStage.setScene(scene);
      primaryStage.show();
    } catch (Exception ex) {
      ex.printStackTrace();
    }
  }

  public static void main(String[] args) {
    launch(args);
  }
}
