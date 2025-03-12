package com.mycompany.train_model;

import com.google.gson.Gson;
import weka.core.*;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.supervised.instance.Resample;
import weka.filters.Filter;
import com.huaban.analysis.jieba.JiebaSegmenter;
import java.text.Normalizer;
import java.util.List;
import java.util.StringTokenizer;

// 演算法
import weka.classifiers.functions.Logistic;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.MultilayerPerceptron;

import com.github.jfasttext.JFastText;
import com.huaban.analysis.jieba.WordDictionary;
import weka.classifiers.meta.FilteredClassifier;
import weka.filters.MultiFilter;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Train_model {

    private static JFastText fastText;
    private static final Set<String> stopWords = new HashSet<>();

    static {
        loadStopWords();
    }

    // ============ 共用工具方法 ============
    /**
     * 文本預處理：全形轉半形、分詞、去除停用詞與特定符號
     */
    private static String preprocessTextCommon(String text) {
        // 1. 全形轉半形
        String normalized = Normalizer.normalize(text, Normalizer.Form.NFKC);

        // 2. 使用 jieba 進行中文分詞
        JiebaSegmenter segmenter = new JiebaSegmenter();
        List<String> tokens = segmenter.sentenceProcess(normalized);
        // 將分詞結果以空格連接成字串
        String tokenized = String.join(" ", tokens);

        // 3. 過濾停用詞及特殊字元（如 %, $, #, +, /, & 等）
        StringTokenizer tokenizer = new StringTokenizer(tokenized);
        StringBuilder filtered = new StringBuilder();
        while (tokenizer.hasMoreTokens()) {
            String word = tokenizer.nextToken();
            if (!stopWords.contains(word)) {
                filtered.append(word).append(" ");
            }
        }
        return filtered.toString().trim();
    }

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
                    return new double[vectorSize]; // 返回零向量，避免 null 指針問題
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
     * 讀取停用詞檔案
     */
    private static void loadStopWords() {
        try (InputStream inputStream = Train_model.class.getClassLoader().getResourceAsStream("stopwords.txt"); BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            while ((line = reader.readLine()) != null) {
                stopWords.add(line.trim());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 匯出 Instances 為 CSV
     */
    private static void exportToCSV(Instances data, String csvPath) throws IOException {
        File outputFile = new File(csvPath);
        if (outputFile.exists()) {
            System.out.println("File already exists: " + csvPath);
            if (!outputFile.delete()) {
                throw new IOException("Unable to delete existing file: " + csvPath);
            }
            System.out.println("Existing file deleted.");
        }
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);
        saver.setFile(outputFile);
        saver.writeBatch();
        System.out.println("Data exported to: " + csvPath);
    }

    /**
     * 使用多執行緒對資料做預處理
     */
    private static Instances preprocessDataWithThreads(Instances data) throws Exception {
        ArrayList<String> classValues = new ArrayList<>();
        for (int i = 0; i < data.numClasses(); i++) {
            classValues.add(data.classAttribute().value(i));
        }
        // 創建 Weka 的特徵屬性 (文本 + 類別標籤)
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("text", (List<String>) null));
        attributes.add(new Attribute("class", classValues));
        Instances processedData = new Instances("ProcessedData", attributes, data.numInstances());
        processedData.setClassIndex(processedData.numAttributes() - 1);

        // 創建多執行緒池，CPU核心數作為執行緒數量
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        List<Future<String>> futures = new ArrayList<>();

        // 提交文本預處理任務到執行緒池
        for (int i = 0; i < data.numInstances(); i++) {
            String originalText = data.instance(i).stringValue(0);
            futures.add(executor.submit(() -> preprocessTextCommon(originalText)));
        }

        // 取得處理結果並填入 Weka Instances
        for (int i = 0; i < futures.size(); i++) {
            String processedText;
            try {
                processedText = futures.get(i).get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
                processedText = "empty";
            }
            if (processedText.isEmpty()) {
                processedText = "empty";
            }
            double[] values = new double[2];
            values[0] = processedData.attribute(0).addStringValue(processedText);
            values[1] = data.instance(i).classValue();
            processedData.add(new DenseInstance(1.0, values));
        }
        // 關閉執行緒池
        executor.shutdown();
        return processedData;
    }

    /**
     * 將處理後的文本向量化
     */
    private static Instances vectorizeData(Instances data, JFastText fastText) {
        int vectorSize = fastText.getVector("示例文本").size();
        System.out.println("vectorSize: " + vectorSize);
        ArrayList<Attribute> attributes = new ArrayList<>();
        //創建 300 維的數值型特徵屬性
        for (int i = 0; i < vectorSize; i++) {
            attributes.add(new Attribute("vec_" + i));
        }
        //最後一個屬性設置為 "class"
        attributes.add(data.classAttribute());
        // 創建 Weka 的 Instances 結構
        Instances vectorizedData = new Instances("VectorizedData", attributes, data.numInstances());
        vectorizedData.setClassIndex(vectorizedData.numAttributes() - 1);// 設置類別屬性索引
        
         // 緩存詞向量，提升運算效率
        ConcurrentHashMap<String, double[]> vectorCache = new ConcurrentHashMap<>();

        // 逐條文本處理與向量化
        for (int i = 0; i < data.numInstances(); i++) {
            String text = data.instance(i).stringValue(0);// 取得文本內容
            // 計算平均詞向量
            double[] vector = computeAverageVector(text, fastText, vectorSize, vectorCache);
            // 將詞向量與類別標籤組合為數值陣列
            double[] instanceValues = Arrays.copyOf(vector, vector.length + 1);
            instanceValues[vector.length] = data.instance(i).classValue();// 設置類別值
            // 新增數據至 Instances 中
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
     * 保存類別標籤到文件
     */
    public static void saveLabelsToFile(Instances data) {
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

    /**
     * 根據演算法名稱選擇對應的分類器
     */
    public static Classifier selectClassifier(String algorithmName) {
        switch (algorithmName.toUpperCase()) {
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
                throw new IllegalArgumentException("Unsupported algorithm: " + algorithmName);
        }
    }

    // ============ 主程式 ============
    public static void main(String[] args) {
        String inputCsvPath = "src/main/resources/inputdata.csv";
        String unifiedModelPath = "src/main/resources/model.model";
        String processedCsvPath = "src/main/resources/processed_data.csv";
        String outputTxtPath = "src/main/resources/data_exploration_results.txt";
        String fastTextModelPath = "D:/NCU/weka/embedding/fasttext_model_300.bin";

        Path path = Paths.get("src/main/resources/userdict.txt");
        WordDictionary.getInstance().loadUserDict(path);

        long startTime = System.currentTimeMillis();
        try {
            // 載入資料
            CSVLoader loader = new CSVLoader();
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
            // 資料探索（類別分布、文本長度、缺失值檢查）
            DataExploration.analyzeClassDistribution(data, outputTxtPath);
            DataExploration.analyzeTextLength(data, outputTxtPath);
            DataExploration.checkMissingValues(data, outputTxtPath);

            // 資料預處理（多執行緒處理）
            Instances processedData = preprocessDataWithThreads(data);
            exportToCSV(processedData, processedCsvPath);

            // 載入 FastText 模型並向量化資料
            fastText = loadFastTextModel(fastTextModelPath);
            Instances vectorizedData = vectorizeData(processedData, fastText);

            // 利用 MultiFilter 組合 Resample 與 ClassBalancer 過濾器
            Resample resample = new Resample();
            resample.setNoReplacement(false); // 有放回抽樣
            resample.setBiasToUniformClass(0.5);// 平衡類別分布
            resample.setSampleSizePercent(150);// 重抽樣比例

            ClassBalancer classBalancer = new ClassBalancer();

            MultiFilter multiFilter = new MultiFilter();
            Filter[] filters = new Filter[2];
            filters[0] = resample;
            filters[1] = classBalancer;
            multiFilter.setFilters(filters);

            // 選擇基礎分類器 (此處以 RF 為例)
            Classifier baseClassifier = selectClassifier("RF");

            // 將過濾器與分類器封裝成 FilteredClassifier
            FilteredClassifier filteredClassifier = new FilteredClassifier();
            filteredClassifier.setFilter(multiFilter);
            filteredClassifier.setClassifier(baseClassifier);

            // 利用交叉驗證評估模型，確保每個 fold 中僅在訓練階段應用過濾器
            Evaluation eval = new Evaluation(vectorizedData);
            int numFolds = 10;
            eval.crossValidateModel(filteredClassifier, vectorizedData, numFolds, new Random(11));

            System.out.println("=== Summary ===");
            System.out.println(eval.toSummaryString());
            System.out.println("\n=== Evaluation Metrics ===");
            System.out.println("Test Set Accuracy: " + eval.pctCorrect() + "%");
            System.out.println("Precision: " + eval.weightedPrecision());
            System.out.println("Recall: " + eval.weightedRecall());
            System.out.println("F-Measure: " + eval.weightedFMeasure());
            System.out.println("ROC Area: " + eval.weightedAreaUnderROC());
            // 使用 Weka 的 Evaluation 顯示混淆矩陣
            System.out.println(eval.toMatrixString("=== Confusion Matrix ==="));

            // 儲存類別標籤
            saveLabelsToFile(data);

            // 最後使用整個資料集訓練模型
            filteredClassifier.buildClassifier(vectorizedData);
            eval.evaluateModel(filteredClassifier, vectorizedData);
            System.out.println("Final Model Accuracy: " + eval.pctCorrect() + "%");

            List<String> classValues = new ArrayList<>();
            for (int i = 0; i < data.numClasses(); i++) {
                classValues.add(data.classAttribute().value(i));
            }

            // 讀取 userdict.txt 內容
            String userDictPath = "src/main/resources/userdict.txt";
            List<String> userDictContent = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new FileReader(userDictPath))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    userDictContent.add(line.trim());
                }
                System.out.println("User dictionary loaded from: " + userDictPath);
            } catch (IOException e) {
                e.printStackTrace();
            }

            UnifiedModel unifiedModel = new UnifiedModel(fastText, filteredClassifier, classValues, userDictContent);
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

        public String predictClass(String inputText) throws Exception {
            return model.predictClass(inputText);
        }
    }

    // ============ UnifiedModel 類別 ============
    public static class UnifiedModel implements Serializable {

        private final Classifier classifier;
        private transient JFastText fastText;
        private final List<String> classValues;
        private final ConcurrentHashMap<String, double[]> vectorCache = new ConcurrentHashMap<>();
        private final List<String> userDictContent; // 新增用於保存詞典內容

        public UnifiedModel(JFastText fastText, Classifier classifier, List<String> classValues, List<String> userDictContent) {
            this.fastText = fastText;
            this.classifier = classifier;
            this.classValues = classValues;
            this.userDictContent = userDictContent; // 初始化詞典內容
        }

        public void setFastText(JFastText fastText) {
            this.fastText = fastText;
        }

        // 加載詞典內容到 Jieba 的 WordDictionary
        public void loadUserDict() {
            if (userDictContent != null) {
                // 創建臨時文件以加載自定義詞典
                try {
                    File tempFile = File.createTempFile("userdict", ".txt");
                    tempFile.deleteOnExit(); // 程式結束時自動刪除
                    try (BufferedWriter writer = new BufferedWriter(new FileWriter(tempFile))) {
                        for (String word : userDictContent) {
                            writer.write(word);
                            writer.newLine();
                        }
                    }
                    Path path = Paths.get(tempFile.getAbsolutePath());
                    WordDictionary.getInstance().loadUserDict(path);
                    System.out.println("User dictionary loaded successfully."+ tempFile.getAbsolutePath() );
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
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
            //System.out.println("inputText: " + inputText);
            //System.out.println("processedText: " + processedData.instance(0).stringValue(0));
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

        public String predictClass(String inputText) throws Exception {
            Instances processedData = preprocessData(inputText);
            System.out.println("inputText: " + inputText);
            System.out.println("processedText: " + processedData.instance(0).stringValue(0));
            double[] vector = calculateAverageVector(processedData);
            Instances instance = createPredictionInstance(vector);
            double predictedClassValue = classifier.classifyInstance(instance.instance(0));
            if (predictedClassValue < 0 || predictedClassValue >= classValues.size()) {
                throw new ArrayIndexOutOfBoundsException("Predicted class value out of range: " + predictedClassValue);
            }
            return classValues.get((int) predictedClassValue);
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
            // 建立與訓練階段相同的屬性結構
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("text", (List<String>) null));
            attributes.add(new Attribute("class", new ArrayList<>(classValues)));
            Instances processedData = new Instances("ProcessedData", attributes, 1);
            processedData.setClassIndex(processedData.numAttributes() - 1);

            String processedText = preprocessTextCommon(data);
            if (processedText.isEmpty()) {
                processedText = "empty";
            }
            double[] values = new double[2];
            values[0] = processedData.attribute(0).addStringValue(processedText);
            // 類別未知，設為缺失值
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
                UnifiedModel model = (UnifiedModel) ois.readObject();
                model.loadUserDict(); // 在加載模型時自動加載詞典
                return model;
            }
        }
    }
}
