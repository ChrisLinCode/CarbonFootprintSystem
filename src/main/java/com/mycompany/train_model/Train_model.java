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
import weka.filters.unsupervised.attribute.Standardize;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;

// 演算法
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
//import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.J48;
import weka.classifiers.meta.AdaBoostM1;

import com.github.jfasttext.JFastText;
import weka.classifiers.meta.FilteredClassifier;
import weka.filters.MultiFilter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.core.OptionHandler;

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
     * 回傳已使用最佳參數 train 且輸出 10-fold CV 成績的 classifier
     */
    public static Classifier selectClassifier(String algorithmName, Instances trainData) throws Exception {
        Classifier cls;

        switch (algorithmName.toUpperCase()) {
            case "RF": {
                // Random Forest: numTrees = 180
                RandomForest rf = new RandomForest();
                ((OptionHandler) rf).setOptions(new String[]{"-I", "180"});
                cls = rf;
                break;
            }
            case "SVM": {
                // SMO: C = 100, kernel = RBF
                SMO smo = new SMO();
                smo.setC(100);
                smo.setKernel(new RBFKernel());
                cls = smo;
                break;
            }
            case "KNN": {
                // IBk: K = 1
                cls = new IBk(1);
                break;
            }
            case "XGB":
            case "XGBOOST": {
                // XGBoost: num_round = 190
                XGBoostClassifierWrapper xgb = new XGBoostClassifierWrapper();
                ((OptionHandler) xgb).setOptions(new String[]{"-num_round", "190"});
                cls = xgb;
                break;
            }
            case "ADA":
            case "ADABOOST": {
                // AdaBoostM1 + J48: iterations = 100
                AdaBoostM1 ab = new AdaBoostM1();
                ab.setClassifier(new J48());
                ab.setNumIterations(100);
                cls = ab;
                break;
            }
            default:
                throw new IllegalArgumentException("Unsupported algorithm: " + algorithmName);
        }

        // 10-fold cross-validation
        Evaluation eval = new Evaluation(trainData);
        eval.crossValidateModel(cls, trainData, 10, new Random(42));

        // 輸出 CV 成績
        System.out.println("\n=== CV Metrics for " + algorithmName + " ===");
        System.out.printf("Accuracy: %.2f%%%n", eval.pctCorrect());
        System.out.printf("Precision: %.4f%n", eval.weightedPrecision());
        System.out.printf("Recall: %.4f%n", eval.weightedRecall());
        System.out.printf("F-Measure: %.4f%n", eval.weightedFMeasure());
        System.out.printf("ROC Area: %.4f%n", eval.weightedAreaUnderROC());
        System.out.println(eval.toMatrixString("=== Confusion Matrix ==="));

        return cls;
    }

    //最佳參數搜索
    public static Classifier tuneWithCV(String classifierName, Instances trainData) throws Exception {
        Classifier tuned;

        if ("SVM".equalsIgnoreCase(classifierName)) {
            // SVM: C ∈ {0.01,0.1,1,10,100}（log scale），kernel ∈ {Poly, RBF}
            Kernel[] kernels = {new PolyKernel(), new RBFKernel()};
            double[] cGrid = {0.01, 0.1, 1, 10, 100};
            double bestAcc = Double.NEGATIVE_INFINITY;
            double bestC = cGrid[0];
            Kernel bestKernel = kernels[0];

            int total = kernels.length * cGrid.length;
            int count = 0;

            for (Kernel k : kernels) {
                for (double c : cGrid) {
                    count++;
                    System.out.printf("SVM tuning: %d/%d (C=%.4f, Kernel=%s)%n",
                            count, total, c, k.getClass().getSimpleName());

                    SMO svm = new SMO();
                    svm.setC(c);
                    svm.setKernel(k);

                    Evaluation cv = new Evaluation(trainData);
                    cv.crossValidateModel(svm, trainData, 10, new Random(42));

                    double acc = cv.pctCorrect();
                    System.out.printf("  → CV Acc = %.2f%%%n", acc);

                    if (acc > bestAcc) {
                        bestAcc = acc;
                        bestC = c;
                        bestKernel = k;
                    }
                }
            }

            System.out.printf("Best SVM → C=%.4f, Kernel=%s (CV Acc=%.2f%%)%n",
                    bestC, bestKernel.getClass().getSimpleName(), bestAcc);

            SMO finalSVM = new SMO();
            finalSVM.setC(bestC);
            finalSVM.setKernel(bestKernel);
            //finalSVM.buildClassifier(trainData);
            tuned = finalSVM;
        } else if ("KNN".equalsIgnoreCase(classifierName)) {
            // 手動搜尋 K ∈ {1…15}
            int bestK = 1;
            double bestAcc = Double.NEGATIVE_INFINITY;

            for (int k = 1; k <= 15; k++) {
                IBk knn = new IBk(k);
                Evaluation cv = new Evaluation(trainData);
                cv.crossValidateModel(knn, trainData, 10, new Random(42));
                double acc = cv.pctCorrect();
                System.out.printf("CV @ K=%2d → Acc = %.2f%%%n", k, acc);
                if (acc > bestAcc) {
                    bestAcc = acc;
                    bestK = k;
                }
            }

            System.out.printf("Best K = %d (CV Acc=%.2f%%)%n", bestK, bestAcc);
            IBk finalKNN = new IBk(bestK);
            //finalKNN.buildClassifier(trainData);
            tuned = finalKNN;

        } else if ("ADA".equalsIgnoreCase(classifierName) || "ADABOOST".equalsIgnoreCase(classifierName)) {
            // AdaBoostM1 + J48 基底分類器，調整迭代次數 I ∈ {10,20,…,100}
            int bestIters = 10;
            double bestAcc = Double.NEGATIVE_INFINITY;
            int total = (100 - 10) / 10 + 1;
            for (int idx = 0; idx < total; idx++) {
                int iters = 10 + idx * 10;
                System.out.printf("AdaBoost tuning: %d/%d (I=%d)%n", idx + 1, total, iters);

                AdaBoostM1 ab = new AdaBoostM1();
                ab.setClassifier(new J48());
                ab.setNumIterations(iters);
                Evaluation cv = new Evaluation(trainData);
                cv.crossValidateModel(ab, trainData, 10, new Random(42));

                double acc = cv.pctCorrect();
                System.out.printf("  → CV Acc = %.2f%%%n", acc);
                if (acc > bestAcc) {
                    bestAcc = acc;
                    bestIters = iters;
                }
            }
            System.out.printf("Best AdaBoost I = %d (CV Acc=%.2f%%)%n", bestIters, bestAcc);

            AdaBoostM1 finalAB = new AdaBoostM1();
            finalAB.setClassifier(new J48());
            finalAB.setNumIterations(bestIters);
            //finalAB.buildClassifier(trainData);
            tuned = finalAB;

        } else if ("RF".equalsIgnoreCase(classifierName)) {
            // 手動搜尋 numTrees ∈ {50,60,…,200}
            int bestTrees = 50;
            double bestAcc = Double.NEGATIVE_INFINITY;
            int totalSteps = (200 - 50) / 10 + 1;

            for (int i = 0; i < totalSteps; i++) {
                int t = 50 + i * 10;
                System.out.printf("RF tuning: %d/%d (numTrees=%d)%n", i + 1, totalSteps, t);

                RandomForest rf = new RandomForest();
                // 用 OptionHandler 設定 -I 樹數
                ((OptionHandler) rf).setOptions(new String[]{"-I", Integer.toString(t)});

                Evaluation cv = new Evaluation(trainData);
                cv.crossValidateModel(rf, trainData, 10, new Random(42));

                double acc = cv.pctCorrect();
                System.out.printf("  → CV Acc = %.2f%%%n", acc);

                if (acc > bestAcc) {
                    bestAcc = acc;
                    bestTrees = t;
                }
            }

            System.out.printf("Best numTrees = %d (CV Acc=%.2f%%)%n", bestTrees, bestAcc);

            // 用最佳樹數訓練最終模型
            RandomForest finalRF = new RandomForest();
            ((OptionHandler) finalRF).setOptions(new String[]{"-I", Integer.toString(bestTrees)});
            //finalRF.buildClassifier(trainData);
            tuned = finalRF;
        } else if ("XGB".equalsIgnoreCase(classifierName) || "XGBOOST".equalsIgnoreCase(classifierName)) {
            // (XGB 部分維持之前的手動搜尋 num_round)
            int bestRounds = 50;
            double bestAcc = Double.NEGATIVE_INFINITY;
            int total = (200 - 50) / 10 + 1;

            for (int i = 0; i < total; i++) {
                int r = 50 + i * 10;
                System.out.printf("XGB tuning: %d/%d (num_round=%d)%n", i + 1, total, r);

                XGBoostClassifierWrapper xgb = new XGBoostClassifierWrapper();
                ((OptionHandler) xgb).setOptions(new String[]{"-num_round", Integer.toString(r)});

                Evaluation cv = new Evaluation(trainData);
                cv.crossValidateModel(xgb, trainData, 10, new Random(42));

                double acc = cv.pctCorrect();
                System.out.printf("  → CV Acc = %.2f%%%n", acc);

                if (acc > bestAcc) {
                    bestAcc = acc;
                    bestRounds = r;
                }
            }

            System.out.printf("Best num_round = %d (CV Acc=%.2f%%)%n", bestRounds, bestAcc);

            XGBoostClassifierWrapper finalXGB = new XGBoostClassifierWrapper();
            ((OptionHandler) finalXGB).setOptions(new String[]{"-num_round", Integer.toString(bestRounds)});
            //finalXGB.buildClassifier(trainData);
            tuned = finalXGB;
        } else {
            throw new IllegalArgumentException("Unsupported classifier: " + classifierName);
        }

        // 10-fold CV on the tuned model
        Evaluation evalCV = new Evaluation(trainData);
        evalCV.crossValidateModel(tuned, trainData, 10, new Random(42));
        System.out.println("\n=== CV Metrics for " + classifierName + " ===");
        System.out.printf("CV Accuracy: %.2f%%\n", evalCV.pctCorrect());
        System.out.printf("CV Precision: %.4f\n", evalCV.weightedPrecision());
        System.out.printf("CV Recall: %.4f\n", evalCV.weightedRecall());
        System.out.printf("CV F-Measure: %.4f\n", evalCV.weightedFMeasure());
        System.out.printf("CV ROC Area: %.4f\n", evalCV.weightedAreaUnderROC());

        // retrain final model on entire trainData
        //tuned.buildClassifier(trainData);
        return tuned;
    }

    // ============ 主程式 ============
    public static void main(String[] args) {
        String inputCsvPath = "src/main/resources/inputdata.csv";
        String unifiedModelPath = "src/main/resources/model.model";
        String outputTxtPath = "src/main/resources/data_exploration_results.txt";
        String fastTextModelPath = "D:/NCU/weka/embedding/fasttext_model_300.bin";

        //long startTime = System.currentTimeMillis();
        try {
            // 載入資料
            CSVLoader loader = new CSVLoader();
            loader.setOptions(new String[]{"-encoding", "UTF-8"});
            loader.setSource(new File(inputCsvPath));
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);
            //System.out.println("Data loaded. Time: " + (System.currentTimeMillis() - startTime) + " ms");

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
            // 檢查 resources 下是否已有預處理結果檔案
            File preprocessedFile = new File("src/main/resources/preprocessData.arff");
            Instances processedData;
            if (preprocessedFile.exists()) {
                // 已有結果，直接讀取
                processedData = DataSource.read(preprocessedFile.getAbsolutePath());
                processedData.setClassIndex(processedData.numAttributes() - 1);
            } else {
                // 無結果，執行分詞並儲存
                processedData = preprocessDataBatch(data);
                ArffSaver saver = new ArffSaver();
                saver.setInstances(processedData);
                saver.setFile(preprocessedFile);
                saver.writeBatch();
            }

            // 載入 FastText 模型並向量化資料
            fastText = loadFastTextModel(fastTextModelPath);
            Instances vectorizedData = vectorizeData(processedData, fastText);

            // 1. 先打亂、再做 stratify（分層）
            vectorizedData.randomize(new Random(42));
            if (vectorizedData.classAttribute().isNominal()) {
                vectorizedData.stratify(10);
            }

            // 2. 用第一支 filter 取出 trainData（90%）
            StratifiedRemoveFolds trainFold = new StratifiedRemoveFolds();
            trainFold.setNumFolds(10);
            trainFold.setSeed(42);
            trainFold.setFold(1);
            trainFold.setInvertSelection(true);            // keep 90%
            trainFold.setInputFormat(vectorizedData);       // **每支 filter 都要呼叫一次**
            Instances trainData = Filter.useFilter(vectorizedData, trainFold);

            // 3. 用第二支 filter 取出 testData（10%）
            StratifiedRemoveFolds testFold = new StratifiedRemoveFolds();
            testFold.setNumFolds(10);
            testFold.setSeed(42);
            testFold.setFold(1);
            testFold.setInvertSelection(false);             // keep only fold #1 (10%)
            testFold.setInputFormat(vectorizedData);       // **重新呼叫一次**
            Instances testData = Filter.useFilter(vectorizedData, testFold);

            // 4. 檢查大小
            System.out.println("Train size: " + trainData.numInstances());
            System.out.println("Test  size: " + testData.numInstances());

            // 建立 MultiFilter
            Standardize standardize = new Standardize();
            ClassBalancer classBalancer = new ClassBalancer();

            MultiFilter multiFilter = new MultiFilter();
            Filter[] filters = new Filter[]{
                standardize, // Step 1: 標準化
                classBalancer // Step 3: 類別平衡
            };
            multiFilter.setFilters(filters);

            
            //選擇是否進行超參數搜索
            int searchFlag = 0;
            String model = "SVM";

            Classifier bestModel;
            if (searchFlag == 1) {
                // 執行超參數搜尋
                bestModel = tuneWithCV(model, trainData);
            } else {
                // 直接回傳已設定好最佳參數的模型
                bestModel = selectClassifier(model, trainData);
            }

            FilteredClassifier filteredClassifier = new FilteredClassifier();
            filteredClassifier.setFilter(multiFilter);
            filteredClassifier.setClassifier(bestModel);

            // 先用訓練資料 trainData 建立 FilteredClassifier
            filteredClassifier.buildClassifier(trainData);

            // 建立 Evaluation 物件（參數用 trainData 以對類別分佈做參考）
            Evaluation evalTest = new Evaluation(trainData);

            // 在 testData 上做評估
            evalTest.evaluateModel(filteredClassifier, testData);

            // 印出各項指標
            System.out.println("\n=== Test Set Evaluation ===");
            System.out.printf("Accuracy: %.2f%%\n", evalTest.pctCorrect());
            System.out.printf("Precision: %.4f\n", evalTest.weightedPrecision());
            System.out.printf("Recall: %.4f\n", evalTest.weightedRecall());
            System.out.printf("F-Measure: %.4f\n", evalTest.weightedFMeasure());
            System.out.printf("ROC Area: %.4f\n", evalTest.weightedAreaUnderROC());

            System.out.println(evalTest.toMatrixString("=== Confusion Matrix ==="));

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

            // Weka instances must provide a value for every attribute, including the class
            // attribute. Previously only the feature vector was supplied, which resulted in
            // an array that was one element too short and triggered an exception during
            // prediction. Here we append a missing value placeholder for the class label so
            // the instance length matches the schema expected by Weka.
            double[] values = Arrays.copyOf(vector, vector.length + 1);
            values[vector.length] = Utils.missingValue();
            instance.add(new DenseInstance(1.0, values));
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
