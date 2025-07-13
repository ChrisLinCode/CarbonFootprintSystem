package com.mycompany.train_model;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import java.util.Enumeration;
import java.util.Vector;
import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * XGBoost classifier wrapper for Weka.
 * Supports setting number of boosting rounds via -num_round option.
 */
public class XGBoostClassifierWrapper extends AbstractClassifier implements OptionHandler {
    private static final long serialVersionUID = 1L;

    /** Number of boosting rounds (trees) */
    protected int numRound = 100;
    protected Booster booster;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // Convert Instances to DMatrix
        DMatrix trainMatrix = convertInstancesToDMatrix(data);

        // Set parameters
        HashMap<String, Object> params = new HashMap<>();
        int numClasses = data.numClasses();
        if (numClasses > 2) {
            params.put("objective", "multi:softprob");
            params.put("num_class", numClasses);
        } else {
            params.put("objective", "binary:logistic");
        }

        // Train booster
        booster = XGBoost.train(trainMatrix, params, numRound, new HashMap<>(), null, null);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        // Convert single instance to DMatrix
        DMatrix dm = convertInstanceToDMatrix(instance);
        float[][] predicts = booster.predict(dm);

        if (predicts[0].length == 1) {
            // Binary classification threshold 0.5
            return predicts[0][0] > 0.5 ? 1.0 : 0.0;
        } else {
            // Multi-class: take argmax
            int maxIndex = 0;
            for (int i = 1; i < predicts[0].length; i++) {
                if (predicts[0][i] > predicts[0][maxIndex]) {
                    maxIndex = i;
                }
            }
            return (double) maxIndex;
        }
    }

    // ===== OptionHandler implementation =====

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> options = new Vector<>();
        options.add(new Option(
            "\tNumber of boosting rounds (trees). Default = 100.\n",
            "num_round", 1, "-num_round <int>"));
        return options.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        // Parse num_round option
        String nr = Utils.getOption("num_round", options);
        if (nr.length() > 0) {
            numRound = Integer.parseInt(nr);
        }
        // Check for remaining unsupported options
        Utils.checkForRemainingOptions(options);
    }

    @Override
    public String[] getOptions() {
        List<String> opts = new ArrayList<>();
        opts.add("-num_round");
        opts.add(Integer.toString(numRound));
        return opts.toArray(new String[0]);
    }

    // ===== Helper methods to convert data =====

    protected DMatrix convertInstancesToDMatrix(Instances data) throws Exception {
        int nRows = data.numInstances();
        int nCols = data.numAttributes() - 1;
        float[] values = new float[nRows * nCols];
        float[] labels = new float[nRows];
        for (int i = 0; i < nRows; i++) {
            Instance inst = data.instance(i);
            for (int j = 0; j < nCols; j++) {
                values[i * nCols + j] = (float) inst.value(j);
            }
            labels[i] = (float) inst.classValue();
        }
        //DMatrix matrix = new DMatrix(values, nRows, nCols);
        DMatrix matrix = new DMatrix(values, nRows, nCols, Float.NaN);
        matrix.setLabel(labels);
        return matrix;
    }

    protected DMatrix convertInstanceToDMatrix(Instance inst) throws Exception {
        int nCols = inst.numAttributes() - 1;
        float[] values = new float[nCols];
        for (int j = 0; j < nCols; j++) {
            values[j] = (float) inst.value(j);
        }
        //return new DMatrix(values, 1, nCols);
        return new DMatrix(values, 1, nCols, Float.NaN);
    }
}
