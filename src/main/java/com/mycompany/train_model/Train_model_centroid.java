package com.mycompany.train_model;

//記得先開啟flask
import com.github.jfasttext.JFastText;
import com.google.gson.Gson;
import weka.core.*;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;

import java.io.*;
import java.text.Normalizer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class Train_model_centroid {

    private static JFastText fastText;

    public static void main(String[] args) {
        // ============= 路徑設定 (請依實際情況調整) =============
        String inputCsvPath = "src/main/resources/inputdata.csv";
        String processedCsvPath = "src/main/resources/processed_data.csv";
        String outputTxtPath = "src/main/resources/data_exploration_results.txt";
        String fastTextModelPath = "D:/NCU/weka/embedding/fasttext_model_300.bin";
        String unifiedModelPath = "src/main/resources/model.model"; // 儲存/載入模型

        try {
            // 1. 載入 CSV
            long startTime = System.currentTimeMillis();
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(inputCsvPath));
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1); // 最後一欄當作 class

            System.out.println("Data loaded. Time: " + (System.currentTimeMillis() - startTime) + " ms");

            // 2. 資料探索輸出 (若需要)
            try (PrintWriter writer = new PrintWriter(new File(outputTxtPath))) {
                writer.print("");
            } catch (IOException e) {
                System.out.println("Failed to clear file: " + e.getMessage());
            }
            DataExploration.analyzeClassDistribution(data, outputTxtPath);
            DataExploration.analyzeTextLength(data, outputTxtPath);
            DataExploration.checkMissingValues(data, outputTxtPath);

            // 3. 單執行緒預處理 (直接用 CKIP 分詞)
            Instances processedData = preprocessData(data);
            exportToCSV(processedData, processedCsvPath);

            // 4. 載入 FastText
            fastText = loadFastTextModel(fastTextModelPath);

            // 5. Cross-Validation
            System.out.println("=== Start 10-fold Cross Validation ===");
            CrossValidationResult cvResult = crossValidateCentroid(processedData, 10, new Random(42));
            System.out.println(cvResult);

            // 6. 建立最終模型(類別重心) + 訓練
            UnifiedModel finalModel = new UnifiedModel();
            finalModel.buildCentroidModel(processedData);
            System.out.println("\nTrain on entire dataset done.");

            // 7. 保存模型
            UnifiedModel.saveModel(finalModel, unifiedModelPath);
            System.out.println("Model saved to: " + unifiedModelPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // ------------------------------------------
    // A. 文本預處理 (改用 CKIPClientAPI 分詞)
    // ------------------------------------------
    private static Instances preprocessData(Instances data) throws Exception {
        // 建立新 Instances: text + class
        ArrayList<String> classValues = new ArrayList<>();
        for (int i = 0; i < data.numClasses(); i++) {
            classValues.add(data.classAttribute().value(i));
        }
        ArrayList<Attribute> attrs = new ArrayList<>();
        attrs.add(new Attribute("text", (List<String>) null));
        attrs.add(new Attribute("class", classValues));
        Instances newData = new Instances("ProcessedData", attrs, data.numInstances());
        newData.setClassIndex(newData.numAttributes() - 1);

        for (int i = 0; i < data.numInstances(); i++) {
            String originalText = data.instance(i).stringValue(0);
            String label = data.instance(i).stringValue(data.classIndex());

            String cleanText = preprocessTextCommon(originalText); // 使用 CKIP 分詞
            if (cleanText.isEmpty()) {
                cleanText = "empty";
            }

            double[] vals = new double[2];
            // text
            vals[0] = newData.attribute(0).addStringValue(cleanText);
            // class
            int labelIndex = classValues.indexOf(label);
            vals[1] = labelIndex;
            newData.add(new DenseInstance(1.0, vals));
        }

        return newData;
    }

    private static String preprocessTextCommon(String text) {
        try {
            // 直接呼叫 CKIPClientAPI 取得分詞結果
            return CKIPClientAPI.segment(text);
        } catch (Exception e) {
            e.printStackTrace();
            // 若 API 呼叫失敗，回傳原始文本（或做適當 fallback）
            return text;
        }
    }

    private static void exportToCSV(Instances data, String csvPath) throws IOException {
        File out = new File(csvPath);
        if (out.exists()) {
            out.delete();
        }
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);
        saver.setFile(out);
        saver.writeBatch();
        System.out.println("Data exported to: " + csvPath);
    }

    // ------------------------------------------
    // B. 讀取 FastText
    // ------------------------------------------
    private static JFastText loadFastTextModel(String modelFilePath) throws IOException {
        File f = new File(modelFilePath);
        if (!f.exists()) {
            throw new FileNotFoundException("FastText model not found: " + modelFilePath);
        }
        JFastText ft = new JFastText();
        ft.loadModel(f.getAbsolutePath());
        System.out.println("FastText model loaded from: " + modelFilePath);
        return ft;
    }

    // ------------------------------------------
    // C. Cross-Validation (移除 userdict 相關參數)
    // ------------------------------------------
    public static CrossValidationResult crossValidateCentroid(Instances data, int folds, Random rnd) throws Exception {
        data.randomize(rnd);
        if (data.classAttribute().isNominal()) {
            data.stratify(folds);
        }
        int numClasses = data.numClasses();
        int[][] confusionMatrix = new int[numClasses][numClasses];

        for (int n = 0; n < folds; n++) {
            Instances train = data.trainCV(folds, n, rnd);
            Instances test = data.testCV(folds, n);

            // 建立類別重心模型
            UnifiedModel model = new UnifiedModel();
            model.buildCentroidModel(train);

            // 預測 test
            for (int i = 0; i < test.numInstances(); i++) {
                String text = test.instance(i).stringValue(0);
                int actualIdx = (int) test.instance(i).classValue();
                String predLabel = model.predictClass(text);

                int predIdx = model.getClassValues().indexOf(predLabel);
                confusionMatrix[actualIdx][predIdx]++;
            }
        }
        return new CrossValidationResult(confusionMatrix, data.classAttribute());
    }

    // ------------------------------------------
    // D. UnifiedModel：類別重心 + 相似度
    // ------------------------------------------
    public static class UnifiedModel implements Serializable {

        private static final long serialVersionUID = 1L;

        // 類別重心 (key=label, val=平均向量)
        private final Map<String, double[]> classCentroids = new HashMap<>();
        // 所有標籤
        private final List<String> classValues = new ArrayList<>();
        // 快取詞向量
        private final ConcurrentHashMap<String, double[]> vectorCache = new ConcurrentHashMap<>();
        // 動態向量維度
        private int vectorSize = -1;

        public UnifiedModel() {
            // 不使用 userdict
        }

        public void buildCentroidModel(Instances data) throws Exception {
            // 1. 收集所有類別標籤
            for (int i = 0; i < data.numClasses(); i++) {
                classValues.add(data.classAttribute().value(i));
            }

            // 2. 動態抓取 fastText 向量維度
            if (Train_model_centroid.fastText == null) {
                throw new IllegalStateException("fastText not loaded yet!");
            }
            this.vectorSize = Train_model_centroid.fastText.getVector("示例文本").size();

            // 3. 累計各類別的詞向量
            Map<String, double[]> sumVectors = new HashMap<>();
            Map<String, Integer> sumCounts = new HashMap<>();

            for (int i = 0; i < data.numInstances(); i++) {
                String text = data.instance(i).stringValue(0);
                int labelIdx = (int) data.instance(i).classValue();
                String label = data.classAttribute().value(labelIdx);

                double[] vec = computeAverageVector(text);
                sumVectors.putIfAbsent(label, new double[vectorSize]);
                sumCounts.putIfAbsent(label, 0);

                double[] current = sumVectors.get(label);
                for (int k = 0; k < vectorSize; k++) {
                    current[k] += vec[k];
                }
                sumVectors.put(label, current);
                sumCounts.put(label, sumCounts.get(label) + 1);
            }

            // 4. 求平均 => 重心
            for (String label : sumVectors.keySet()) {
                double[] sums = sumVectors.get(label);
                int count = sumCounts.get(label);
                if (count > 0) {
                    for (int k = 0; k < sums.length; k++) {
                        sums[k] /= count;
                    }
                    classCentroids.put(label, sums);
                }
            }
        }

        // 預測 (帶 quantity)
        public String predict(String inputText, int quantity) throws Exception {
            String predictedLabel = predictClass(inputText);
            Map<String, Object> jsonResult = new LinkedHashMap<>();
            jsonResult.put("name", inputText);
            jsonResult.put("quantity", quantity);
            jsonResult.put("path", predictedLabel);
            return new Gson().toJson(jsonResult);
        }

        // 預測類別標籤
        public String predictClass(String inputText) throws Exception {
            String cleanText = preprocessTextCommon(inputText);
            if (cleanText.isEmpty()) {
                cleanText = "empty";
            }
            double[] vec = computeAverageVector(cleanText);

            String bestLabel = null;
            double bestSim = -Double.MAX_VALUE;
            for (Map.Entry<String, double[]> entry : classCentroids.entrySet()) {
                String label = entry.getKey();
                double[] center = entry.getValue();
                double sim = cosineSimilarity(vec, center);
                if (sim > bestSim) {
                    bestSim = sim;
                    bestLabel = label;
                }
            }
            return bestLabel;
        }

        public List<String> getClassValues() {
            return classValues;
        }

        private double[] computeAverageVector(String text) {
            String[] tokens = text.split("\\s+");
            double[] avg = new double[vectorSize];
            int count = 0;

            for (String token : tokens) {
                double[] wv = vectorCache.computeIfAbsent(token, t -> {
                    List<Float> list = Train_model_centroid.fastText.getVector(t);
                    if (list == null || list.isEmpty()) {
                        return new double[vectorSize]; // 0向量
                    }
                    double[] arr = new double[list.size()];
                    for (int i = 0; i < list.size(); i++) {
                        arr[i] = list.get(i);
                    }
                    return arr;
                });
                for (int i = 0; i < vectorSize; i++) {
                    avg[i] += wv[i];
                }
                count++;
            }
            if (count > 0) {
                for (int i = 0; i < vectorSize; i++) {
                    avg[i] /= count;
                }
            }
            return avg;
        }

        private double cosineSimilarity(double[] v1, double[] v2) {
            double dot = 0, normA = 0, normB = 0;
            for (int i = 0; i < v1.length; i++) {
                dot += v1[i] * v2[i];
                normA += v1[i] * v1[i];
                normB += v2[i] * v2[i];
            }
            return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
        }

        public static void saveModel(UnifiedModel model, String filePath) throws IOException {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
                oos.writeObject(model);
            }
        }

        public static UnifiedModel loadModel(String filePath) throws IOException, ClassNotFoundException {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
                return (UnifiedModel) ois.readObject();
            }
        }

        // 改用 CKIP 分詞 (不使用停用詞與 userdict)
        private static String preprocessTextCommon(String text) {
            try {
                return CKIPClientAPI.segment(text);
            } catch (Exception e) {
                e.printStackTrace();
                return text;
            }
        }
    }

    // ------------------------------------------
    // E. CrossValidationResult (自行計算指標 + 顯示標題列)
    // ------------------------------------------
    public static class CrossValidationResult {

        private final int[][] confusionMatrix;
        private final Attribute classAttr;

        public CrossValidationResult(int[][] confusionMatrix, Attribute classAttr) {
            this.confusionMatrix = confusionMatrix;
            this.classAttr = classAttr;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            int numClasses = confusionMatrix.length;

            // 建立短標籤 (a, b, c, ...)
            List<String> shortLabels = new ArrayList<>();
            for (int i = 0; i < numClasses; i++) {
                // 超過 26 個類別時可自行擴充
                char labelChar = (char) ('a' + i);
                shortLabels.add(String.valueOf(labelChar));
            }

            // 計算整體統計
            int total = 0, correct = 0;
            int[] tp_fp_sum = new int[numClasses];
            int[] tp_fn_sum = new int[numClasses];

            for (int i = 0; i < numClasses; i++) {
                for (int j = 0; j < numClasses; j++) {
                    int val = confusionMatrix[i][j];
                    total += val;
                    tp_fp_sum[j] += val;
                    tp_fn_sum[i] += val;
                    if (i == j) {
                        correct += val;
                    }
                }
            }
            double accuracy = 100.0 * correct / total;

            // 印出總結
            sb.append("=== Cross Validation Result ===\n");
            sb.append("Total Instances: ").append(total).append("\n");
            sb.append(String.format("Overall Accuracy: %.2f%%\n\n", accuracy));

            double weightedPrecision = 0, weightedRecall = 0, weightedF1 = 0;
            for (int c = 0; c < numClasses; c++) {
                int tp = confusionMatrix[c][c];
                int fp = tp_fp_sum[c] - tp;
                int fn = tp_fn_sum[c] - tp;
                int support = tp_fn_sum[c];

                double precision = (tp + fp == 0) ? 0 : (double) tp / (tp + fp);
                double recall = (tp + fn == 0) ? 0 : (double) tp / (tp + fn);
                double f1 = (precision + recall == 0) ? 0 : (2 * precision * recall / (precision + recall));
                double weight = (double) support / total;

                weightedPrecision += precision * weight;
                weightedRecall += recall * weight;
                weightedF1 += f1 * weight;
            }

            sb.append(String.format("Weighted Precision: %.3f\n", weightedPrecision));
            sb.append(String.format("Weighted Recall   : %.3f\n", weightedRecall));
            sb.append(String.format("Weighted F1-Score : %.3f\n\n", weightedF1));

            // 顯示混淆矩陣
            sb.append("=== Confusion Matrix ===\n");
            // 先印出空白 + 上方標題行 (a, b, c, ...)
            sb.append(String.format("%10s", ""));  // 預留空間
            for (int c = 0; c < numClasses; c++) {
                sb.append(String.format("%5s", shortLabels.get(c)));
            }
            sb.append("   <-- classified as\n");

            // 每一行先印 row label ("a=衣/上身類") 再印 matrix 數值
            for (int i = 0; i < numClasses; i++) {
                String rowLabel = String.format("%s=%s", shortLabels.get(i), classAttr.value(i));
                sb.append(String.format("%-10s", rowLabel));  // 左對齊
                for (int j = 0; j < numClasses; j++) {
                    sb.append(String.format("%5d", confusionMatrix[i][j]));
                }
                sb.append("\n");
            }

            return sb.toString();
        }
    }

    // ------------------------------------------
    // F. PredictionAPI (若需要對外提供)
    // ------------------------------------------
    public static class PredictionAPI {

        private UnifiedModel model;

        public PredictionAPI(String modelPath, String fastTextPath) throws IOException, ClassNotFoundException {
            // 載入類別重心模型
            this.model = UnifiedModel.loadModel(modelPath);
            // 載入 fastText
            Train_model_centroid.fastText = loadFastTextModel(fastTextPath);
        }

        // 只保留 predict(String inputText, int quantity)
        public String predict(String inputText, int quantity) throws Exception {
            return model.predict(inputText, quantity);
        }
    }
}
