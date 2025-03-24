package com.mycompany.train_model;
//bert

import com.google.gson.Gson;
import weka.core.*;
import weka.core.converters.CSVLoader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.supervised.instance.Resample;
import weka.core.converters.CSVSaver;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

// 如果您有其他演算法, 可按需保留/刪除
import weka.classifiers.functions.Logistic;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.MultilayerPerceptron;

//flask
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class Train_model_bert {


    // CSV 檔路徑
    private static final String INPUT_CSV_PATH = "src/main/resources/inputdata.csv";
    private static final String OUTPUT_MODEL_PATH = "src/main/resources/model.model"; // 最終模型
    private static final String EXPLORATION_TXT_PATH = "src/main/resources/data_exploration_results.txt";
    //private static final String PROCESSED_CSV_PATH = "src/main/resources/processed_data_bert.csv"; // 若不需要即可刪除

    public static void main(String[] args) {
        try {
            // ========== (2) 載入資料 ==========
            Instances data = loadData(INPUT_CSV_PATH);
            System.out.println("Data loaded. numInstances=" + data.numInstances());
            data.setClassIndex(data.numAttributes() - 1); // 最後一欄當 class label

            // 清空探索結果檔（若保留 DataExploration）
            try (PrintWriter writer = new PrintWriter(new File(EXPLORATION_TXT_PATH))) {
                writer.print("");
            } catch (IOException e) {
                e.printStackTrace();
            }

            // ========== (3) 資料探索（若需要） ==========
            DataExploration.analyzeClassDistribution(data, EXPLORATION_TXT_PATH);
            DataExploration.analyzeTextLength(data, EXPLORATION_TXT_PATH);
            DataExploration.checkMissingValues(data, EXPLORATION_TXT_PATH);

            // 如果不需匯出 CSV，可刪除以下註解
            // exportToCSV(data, PROCESSED_CSV_PATH);
            // ========== (4) BERT 向量化 (批量) ==========
            Instances vectorizedData = vectorizeDataWithBert(data);
            System.out.println("Vectorized data: " + vectorizedData.numInstances() + " instances, "
                    + (vectorizedData.numAttributes() - 1) + " embedding dims");

            // ========== (5) Resample + ClassBalancer + FilteredClassifier ==========
            Resample resample = new Resample();
            resample.setNoReplacement(false);
            resample.setBiasToUniformClass(0.5);
            resample.setSampleSizePercent(150);

            ClassBalancer classBalancer = new ClassBalancer();

            MultiFilter multiFilter = new MultiFilter();
            Filter[] filters = new Filter[2];
            filters[0] = resample;
            filters[1] = classBalancer;
            multiFilter.setFilters(filters);

            Classifier baseClassifier = selectClassifier("RF"); // 也可換成 DT, NB, SVM, etc.

            FilteredClassifier filteredClassifier = new FilteredClassifier();
            filteredClassifier.setFilter(multiFilter);
            filteredClassifier.setClassifier(baseClassifier);

            // ========== (6) 交叉驗證評估 ==========
            Evaluation eval = new Evaluation(vectorizedData);
            eval.crossValidateModel(filteredClassifier, vectorizedData, 10, new Random(42));

            System.out.println("=== Cross-Validation ===");
            System.out.println(eval.toSummaryString());
            System.out.println("Accuracy: " + eval.pctCorrect());
            System.out.println("Precision: " + eval.weightedPrecision());
            System.out.println("Recall: " + eval.weightedRecall());
            System.out.println("F1: " + eval.weightedFMeasure());
            System.out.println("AUC: " + eval.weightedAreaUnderROC());
            System.out.println(eval.toMatrixString("=== Confusion Matrix ==="));

            // ========== (7) 用整個資料集訓練最終模型 ==========
            filteredClassifier.buildClassifier(vectorizedData);

            // 再評估一次 (在整個資料集上)
            eval.evaluateModel(filteredClassifier, vectorizedData);
            System.out.println("Final Model Accuracy (on full data): " + eval.pctCorrect() + "%");

            // ========== (8) 儲存類別標籤、儲存模型 ==========
            saveLabelsToFile(data);

            List<String> classValues = new ArrayList<>();
            for (int i = 0; i < data.numClasses(); i++) {
                classValues.add(data.classAttribute().value(i));
            }

            // 建立並儲存 UnifiedModelBert
            UnifiedModelBert unifiedModel = new UnifiedModelBert(filteredClassifier, classValues);
            UnifiedModelBert.saveModel(unifiedModel, OUTPUT_MODEL_PATH);

            System.out.println("Model saved to: " + OUTPUT_MODEL_PATH);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // ========== 讀取 CSV ==========
    private static Instances loadData(String csvPath) throws IOException {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csvPath));
        Instances data = loader.getDataSet();
        return data;
    }

    // ========== BERT 向量化 (批量) ==========
    private static Instances vectorizeDataWithBert(Instances data) throws Exception {
        // 1) 收集所有文本
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < data.numInstances(); i++) {
            texts.add(data.instance(i).stringValue(0));  // 第 0 欄是文本
        }

        // 2) 一次呼叫 Python, 取得所有 embeddings
        List<double[]> allEmbeddings = getBERTEmbeddings(texts);

        // 3) 建立 Weka 屬性
        int vectorSize = allEmbeddings.get(0).length;
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < vectorSize; i++) {
            attributes.add(new Attribute("vec_" + i));
        }
        // 加入 class
        attributes.add(data.classAttribute());

        Instances vectorizedData = new Instances("VectorizedData", attributes, data.numInstances());
        vectorizedData.setClassIndex(vectorizedData.numAttributes() - 1);

        // 4) 將 embedding 與 class 組合
        for (int i = 0; i < data.numInstances(); i++) {
            double[] embedding = allEmbeddings.get(i);
            double[] instanceVals = Arrays.copyOf(embedding, embedding.length + 1);
            instanceVals[embedding.length] = data.instance(i).classValue();
            vectorizedData.add(new DenseInstance(1.0, instanceVals));
        }
        return vectorizedData;
    }

    private static List<double[]> getBERTEmbeddings(List<String> texts) throws Exception {
        HttpClient client = HttpClient.newHttpClient();
        Map<String, Object> payload = new HashMap<>();
        payload.put("texts", texts);
        String jsonPayload = new Gson().toJson(payload);

        // 設定呼叫的 URL 為部署的 Flask BERT API 端點
        HttpRequest request = HttpRequest.newBuilder()
                .uri(new URI("http://localhost:5001/embed"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                .build();

        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() == 200) {
            Map<?, ?> map = new Gson().fromJson(response.body(), Map.class);
            Object embeddingsObj = map.get("embeddings");
            if (embeddingsObj instanceof List) {
                List<?> embList = (List<?>) embeddingsObj;
                List<double[]> embeddings = new ArrayList<>();
                for (Object embObj : embList) {
                    List<?> innerList = (List<?>) embObj;
                    double[] vector = new double[innerList.size()];
                    for (int i = 0; i < innerList.size(); i++) {
                        vector[i] = ((Number) innerList.get(i)).doubleValue();
                    }
                    embeddings.add(vector);
                }
                return embeddings;
            } else {
                throw new Exception("Unexpected response structure: " + response.body());
            }
        } else {
            throw new Exception("Error calling BERT API: " + response.body());
        }
    }

    // ========== 選擇演算法 ==========
    private static Classifier selectClassifier(String algoName) {
        switch (algoName.toUpperCase()) {
            case "LR":
                Logistic logistic = new Logistic();
                logistic.setDebug(true);
                return logistic;
            case "NB":
                return new NaiveBayes();
            case "DT":
                return new J48();
            case "RF":
                return new RandomForest();
            case "KNN":
                return new IBk();
            case "SVM":
                return new SMO();
            case "MLP":
                return new MultilayerPerceptron();
            case "XGB":
                return new XGBoostClassifierWrapper();
            default:
                throw new IllegalArgumentException("Unsupported algorithm: " + algoName);
        }
    }

    // ========== 儲存類別標籤到檔案 ==========
    private static void saveLabelsToFile(Instances data) {
        String filePath = "src/main/resources/labels.txt";
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (int i = 0; i < data.numClasses(); i++) {
                writer.write(data.classAttribute().value(i));
                writer.newLine();
            }
            System.out.println("Labels saved to: " + filePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    // ============ Prediction API ============
    public static class PredictionAPI {

        private UnifiedModelBert model;

        public PredictionAPI(String unifiedModelPath, String fastTextModelPath)
                throws IOException, ClassNotFoundException {
            this.model = UnifiedModelBert.loadModel(unifiedModelPath);
        }

        public String predict(String inputText, int quantity) throws Exception {
            return model.predict(inputText, quantity);
        }

        public String predictClass(String inputText) throws Exception {
            return model.predictClass(inputText);
        }
    }

    // ========== UnifiedModelBert ==========
    public static class UnifiedModelBert implements Serializable {

        private final Classifier classifier;
        private final List<String> classValues;

        public UnifiedModelBert(Classifier classifier, List<String> classValues) {
            this.classifier = classifier;
            this.classValues = classValues;
        }

        /**
         * 單純預測 class label
         */
        public String predictClass(String text) throws Exception {
            double[] embedding = singleBERTEmbedding(text);
            Instances inst = createPredictionInstance(embedding);
            double pred = classifier.classifyInstance(inst.firstInstance());
            if (pred < 0 || pred >= classValues.size()) {
                return "UNKNOWN";
            }
            return classValues.get((int) pred);
        }

        /**
         * 若要回傳 JSON 或自定義格式
         */
        public String predict(String text, int quantity) throws Exception {
            String predictedClass = predictClass(text);

            Map<String, Object> jsonResult = new LinkedHashMap<>();
            jsonResult.put("name", text);
            jsonResult.put("quantity", quantity);
            jsonResult.put("class", predictedClass);

            return new com.google.gson.Gson().toJson(jsonResult);
        }

        private double[] singleBERTEmbedding(String text) throws Exception {
            List<String> texts = Collections.singletonList(text);
            List<double[]> embeddings = getBERTEmbeddings(texts);
            if (embeddings.isEmpty()) {
                return new double[0];
            }
            return embeddings.get(0);
        }

        private Instances createPredictionInstance(double[] embedding) {
            ArrayList<Attribute> attrs = new ArrayList<>();
            for (int i = 0; i < embedding.length; i++) {
                attrs.add(new Attribute("vec_" + i));
            }
            attrs.add(new Attribute("class", new ArrayList<>(classValues)));

            Instances data = new Instances("PredictionInstance", attrs, 1);
            data.setClassIndex(data.numAttributes() - 1);

            double[] instanceVals = Arrays.copyOf(embedding, embedding.length + 1);
            instanceVals[embedding.length] = Utils.missingValue(); // 預測階段, 無 class label
            data.add(new DenseInstance(1.0, instanceVals));

            return data;
        }

        public static void saveModel(UnifiedModelBert model, String filePath) throws IOException {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
                oos.writeObject(model);
                System.out.println("BERT-based model saved to: " + filePath);
            }
        }

        public static UnifiedModelBert loadModel(String filePath) throws IOException, ClassNotFoundException {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
                return (UnifiedModelBert) ois.readObject();
            }
        }
    }
}
