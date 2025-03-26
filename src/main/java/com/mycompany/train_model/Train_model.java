package com.mycompany.train_model;
//CKIP+fasttext

import com.google.gson.Gson;
import weka.core.*;
import weka.core.converters.CSVLoader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.supervised.instance.Resample;
import weka.filters.Filter;
import java.util.List;

// 演算法
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import com.github.jfasttext.JFastText;
import weka.classifiers.meta.FilteredClassifier;
import weka.filters.MultiFilter;

public class Train_model {

    private static JFastText fastText;

    /**
     * 利用 fastText 與緩存計算輸入文本的平均詞向量
     */
    private static double[] computeAverageVector(String text, JFastText fastText, int vectorSize,
            ConcurrentHashMap<String, double[]> cache) {
        double[] avgVector = new double[vectorSize];
        String[] tokens = text.split(" ");
        int count = 0;
        for (String token : tokens) {
            double[] wordVector = cache.computeIfAbsent(token, t -> {
                List<Float> vecList = fastText.getVector(t);
                if (vecList == null || vecList.isEmpty()) {
                    return new double[vectorSize];
                }
                double[] arr = new double[vecList.size()];
                for (int j = 0; j < vecList.size(); j++) {
                    arr[j] = vecList.get(j);
                }
                return arr;
            });
            if (wordVector == null) {
                continue;
            }
            for (int j = 0; j < wordVector.length; j++) {
                avgVector[j] += wordVector[j];
            }
            count++;
        }
        if (count > 0) {
            for (int j = 0; j < avgVector.length; j++) {
                avgVector[j] /= count;
            }
        }
        return avgVector;
    }

    /**
     * 批量預處理：利用 batch 分詞一次處理所有文本，並建立新的 Instances
     */
    private static Instances preprocessDataBatch(Instances data) throws Exception {
        ArrayList<String> classValues = new ArrayList<>();
        for (int i = 0; i < data.numClasses(); i++) {
            classValues.add(data.classAttribute().value(i));
        }
        // 建立 Weka 特徵屬性 (文本 + 類別)
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("text", (List<String>) null));
        attributes.add(new Attribute("class", classValues));
        Instances processedData = new Instances("ProcessedData", attributes, data.numInstances());
        processedData.setClassIndex(processedData.numAttributes() - 1);

        // 收集所有原始文本
        List<String> originalTexts = new ArrayList<>();
        for (int i = 0; i < data.numInstances(); i++) {
            originalTexts.add(data.instance(i).stringValue(0));
        }
        // 使用批量分詞
        List<String> processedTexts = CKIPClientAPI.segmentBatch(originalTexts);
        for (String txt : processedTexts) {
            System.out.println("分詞結果: " + txt);
        }

        // 將分詞結果寫入 Instances
        for (int i = 0; i < processedTexts.size(); i++) {
            String processedText = processedTexts.get(i);
            if (processedText.isEmpty()) {
                processedText = "empty";
            }
            double[] values = new double[2];
            values[0] = processedData.attribute(0).addStringValue(processedText);
            values[1] = data.instance(i).classValue();
            processedData.add(new DenseInstance(1.0, values));
        }
        return processedData;
    }

    /**
     * 將處理後的文本向量化
     */
    private static Instances vectorizeData(Instances data, JFastText fastText) {
        int vectorSize = fastText.getVector("示例文本").size();
        System.out.println("vectorSize: " + vectorSize);
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < vectorSize; i++) {
            attributes.add(new Attribute("vec_" + i));
        }
        attributes.add(data.classAttribute());
        Instances vectorizedData = new Instances("VectorizedData", attributes, data.numInstances());
        vectorizedData.setClassIndex(vectorizedData.numAttributes() - 1);

        ConcurrentHashMap<String, double[]> vectorCache = new ConcurrentHashMap<>();
        for (int i = 0; i < data.numInstances(); i++) {
            String text = data.instance(i).stringValue(0);
            double[] vector = computeAverageVector(text, fastText, vectorSize, vectorCache);
            double[] instanceValues = Arrays.copyOf(vector, vector.length + 1);
            instanceValues[vector.length] = data.instance(i).classValue();
            vectorizedData.add(new DenseInstance(1.0, instanceValues));
        }
        return vectorizedData;
    }

    /**
     * 載入 FastText 模型
     */
    private static JFastText loadFastTextModel(String modelFilePath) throws IOException {
        File modelFile = new File(modelFilePath);
        if (!modelFile.exists()) {
            throw new FileNotFoundException("FastText model file not found: " + modelFilePath);
        }
        JFastText ft = new JFastText();
        ft.loadModel(modelFile.getAbsolutePath());
        System.out.println("FastText model loaded from: " + modelFilePath);
        return ft;
    }

    /**
     * 根據演算法名稱選擇分類器
     */
    public static Classifier selectClassifier(String algorithmName) {
        switch (algorithmName.toUpperCase()) {
            case "RF":
                return new RandomForest();
            case "KNN":
                return new IBk();
            case "SVM":
                return new SMO();
            case "XGB":
                return new XGBoostClassifierWrapper();
            default:
                throw new IllegalArgumentException("Unsupported algorithm: " + algorithmName);
        }
    }

    // ============ 主程式 ============
    public static void main(String[] args) {
        String inputCsvPath = "src/main/resources/inputdata.csv";
        String unifiedModelPath = "src/main/resources/model.model";
        String outputTxtPath = "src/main/resources/data_exploration_results.txt";
        String fastTextModelPath = "D:/NCU/weka/embedding/fasttext_model_300.bin";

        long startTime = System.currentTimeMillis();
        try {
            // 載入資料
            CSVLoader loader = new CSVLoader();
            loader.setOptions(new String[]{"-encoding", "UTF-8"});
            loader.setSource(new File(inputCsvPath));
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            System.out.println("Data loaded. Time: " + (System.currentTimeMillis() - startTime) + " ms");

            // 清空結果輸出檔案內容
            try (PrintWriter writer = new PrintWriter(new File(outputTxtPath))) {
                writer.print("");
            } catch (IOException e) {
                System.out.println("Failed to clear file: " + e.getMessage());
            }
            // 資料探索（此處呼叫外部 DataExploration 方法進行檢查）
            DataExploration.analyzeClassDistribution(data, outputTxtPath);
            DataExploration.analyzeTextLength(data, outputTxtPath);
            DataExploration.checkMissingValues(data, outputTxtPath);

            // 使用批量預處理一次處理所有文本
            Instances processedData = preprocessDataBatch(data);

            // 載入 FastText 模型並向量化資料
            fastText = loadFastTextModel(fastTextModelPath);
            Instances vectorizedData = vectorizeData(processedData, fastText);

            // 建立 MultiFilter
            Resample resample = new Resample();
            resample.setNoReplacement(false);
            resample.setBiasToUniformClass(0.5);
            resample.setSampleSizePercent(200);

            ClassBalancer classBalancer = new ClassBalancer();

            MultiFilter multiFilter = new MultiFilter();
            Filter[] filters = new Filter[2];
            filters[0] = resample;
            filters[1] = classBalancer;
            multiFilter.setFilters(filters);

            // 選擇分類器 (此處以 RF 為例)
            Classifier baseClassifier = selectClassifier("SVM");

            FilteredClassifier filteredClassifier = new FilteredClassifier();
            filteredClassifier.setFilter(multiFilter);
            filteredClassifier.setClassifier(baseClassifier);

            // 交叉驗證 (確保折數不大於資料數量)
            int numFolds = 5;
            Evaluation eval = new Evaluation(vectorizedData);
            eval.crossValidateModel(filteredClassifier, vectorizedData, numFolds, new Random(42));

            System.out.println("=== Summary ===");
            System.out.println(eval.toSummaryString());
            System.out.println("\n=== Evaluation Metrics ===");
            System.out.println("Test Set Accuracy: " + eval.pctCorrect() + "%");
            System.out.println("Precision: " + eval.weightedPrecision());
            System.out.println("Recall: " + eval.weightedRecall());
            System.out.println("F-Measure: " + eval.weightedFMeasure());
            System.out.println("ROC Area: " + eval.weightedAreaUnderROC());
            System.out.println(eval.toMatrixString("=== Confusion Matrix ==="));

            // 用全部資料訓練模型
            filteredClassifier.buildClassifier(vectorizedData);
            eval.evaluateModel(filteredClassifier, vectorizedData);
            System.out.println("Final Model Accuracy: " + eval.pctCorrect() + "%");

            ArrayList<String> classValues = new ArrayList<>();
            for (int i = 0; i < data.numClasses(); i++) {
                classValues.add(data.classAttribute().value(i));
            }

            UnifiedModel unifiedModel = new UnifiedModel(fastText, filteredClassifier, classValues);
            UnifiedModel.saveModel(unifiedModel, unifiedModelPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // ============ Prediction API ============
    public static class PredictionAPI {

        private UnifiedModel model;

        public PredictionAPI(String unifiedModelPath, String fastTextModelPath)
                throws IOException, ClassNotFoundException {
            this.model = UnifiedModel.loadModel(unifiedModelPath);
            this.model.setFastText(loadFastTextModel(fastTextModelPath));
        }

        public String predict(String inputText, int quantity) throws Exception {
            return model.predict(inputText, quantity);
        }
    }

    // ============ UnifiedModel 類別 ============
    public static class UnifiedModel implements Serializable {

        private final Classifier classifier;
        private transient JFastText fastText;
        private final List<String> classValues;
        private final ConcurrentHashMap<String, double[]> vectorCache = new ConcurrentHashMap<>();

        public UnifiedModel(JFastText fastText, Classifier classifier, List<String> classValues) {
            this.fastText = fastText;
            this.classifier = classifier;
            this.classValues = classValues;
        }

        public void setFastText(JFastText fastText) {
            this.fastText = fastText;
        }

        /**
         * 利用共用方法計算輸入文本平均向量
         */
        private double[] calculateAverageVector(Instances processedData) {
            int vectorSize = fastText.getVector("示例文本").size();
            String text = processedData.instance(0).stringValue(0);
            return computeAverageVector(text, fastText, vectorSize, vectorCache);
        }

        public String predict(String inputText, int quantity) throws Exception {
            Instances processedData = preprocessData(inputText);
            double[] vector = calculateAverageVector(processedData);
            Instances instance = createPredictionInstance(vector);
            double predictedClassValue = classifier.classifyInstance(instance.instance(0));
            String predictedClass = classValues.get((int) predictedClassValue);

            Map<String, Object> jsonResult = new LinkedHashMap<>();
            jsonResult.put("name", inputText);
            jsonResult.put("quantity", quantity);
            jsonResult.put("path", predictedClass);
            return new Gson().toJson(jsonResult);
        }

        private Instances createPredictionInstance(double[] vector) throws Exception {
            ArrayList<Attribute> attributes = new ArrayList<>();
            for (int i = 0; i < vector.length; i++) {
                attributes.add(new Attribute("vec_" + i));
            }
            attributes.add(new Attribute("class", new ArrayList<>(classValues)));
            Instances instance = new Instances("PredictionInstance", attributes, 1);
            instance.setClassIndex(vector.length);
            instance.add(new DenseInstance(1.0, vector));
            return instance;
        }

        private Instances preprocessData(String data) throws Exception {
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("text", (List<String>) null));
            attributes.add(new Attribute("class", new ArrayList<>(classValues)));
            Instances processedData = new Instances("ProcessedData", attributes, 1);
            processedData.setClassIndex(processedData.numAttributes() - 1);

            String processedText = CKIPClientAPI.segment(data);
            if (processedText.isEmpty()) {
                processedText = "empty";
            }
            double[] values = new double[2];
            values[0] = processedData.attribute(0).addStringValue(processedText);
            values[1] = Utils.missingValue();
            processedData.add(new DenseInstance(1.0, values));
            return processedData;
        }

        public static void saveModel(UnifiedModel model, String filePath) throws IOException {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
                oos.writeObject(model);
                System.out.println("Unified model saved to: " + filePath);
            }
        }

        public static UnifiedModel loadModel(String filePath) throws IOException, ClassNotFoundException {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
                return (UnifiedModel) ois.readObject();
            }
        }
    }
}
