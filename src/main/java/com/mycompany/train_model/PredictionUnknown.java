package com.mycompany.train_model;

import java.io.*;
import java.util.*;
//選擇要用哪個版本
import com.mycompany.train_model.Train_model.PredictionAPI; 

public class PredictionUnknown {

    public static void main(String[] args) {
        // 文件路徑
        String inputCsvPath = "src/main/resources/invoice_items.csv"; // 未標註數據文件
        String modelPath = "src/main/resources/model.model"; // 訓練好的模型文件
        String fastTextModelPath = "D:/NCU/weka/embedding/fasttext_model_300.bin";

        try {

            PredictionAPI api = new PredictionAPI(modelPath,fastTextModelPath);

            // 逐行讀取並調用
            try (BufferedReader reader = new BufferedReader(new FileReader(inputCsvPath))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.trim().isEmpty()) {
                        continue; // 跳過空行
                    }

                    String predictedClass = api.predict(line,1);
                    System.out.println("result: " + predictedClass);
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }


}
