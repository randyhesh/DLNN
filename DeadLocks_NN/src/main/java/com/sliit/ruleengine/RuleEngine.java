package com.sliit.ruleengine;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.Rule;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Rule Engine for Genereate Policy Signatures.
 */
public class RuleEngine {

    private Log log = LogFactory.getLog(RuleEngine.class);
    File csvFile;
    CSVLoader csv;
    private JRip ruleMoldel;
    private Instances instances;

    public RuleEngine(String filePath){

        try {
            csvFile = new File(filePath);
            csv = new CSVLoader();
            csv.setSource(csvFile);
        } catch (IOException e) {

            log.error("Error occurred:"+e.getLocalizedMessage());
        }
    }

    public boolean geneateModel(String savePath,boolean save){

        boolean status = false;
        try {

            instances = csv.getDataSet();
            if(!save){

                loadSaveModel(savePath,save);
            }
            else {

                ruleMoldel = new JRip();
            }
            instances.setClassIndex(instances.numAttributes()-1);
            ruleMoldel.buildClassifier(instances);
            loadSaveModel(savePath,true);
            status = true;
        } catch (IOException e) {
            status = false;
            log.error("Error occurred:"+e.getLocalizedMessage());
        } catch (Exception e) {
            status = false;
            log.error("Error occurred:"+e.getLocalizedMessage());
        }
        return status;
    }

    public Map<String,String> evaluateModel(){

        Map<String,String> evaluationSummary = new HashMap<String, String>();
        try {

            instances.setClassIndex(instances.numAttributes()-1);
            Evaluation evaluation = new Evaluation(instances);
            evaluation.evaluateModel(ruleMoldel,instances);
            ArrayList<Rule> rulesList = ruleMoldel.getRuleset();
            String rules = ruleMoldel.toString();
            evaluationSummary.put("rules",rules);
            evaluationSummary.put("summary",evaluation.toSummaryString());
            evaluationSummary.put("confusion_matrix",evaluation.toMatrixString());
        } catch (Exception e) {

            log.error("Error occurred:"+e.getLocalizedMessage());
        }
        return  evaluationSummary;
    }

    public String predictionResult(String filePath) throws Exception {

        File testPath = new File(filePath);
        CSVLoader loader = new CSVLoader();
        loader.setSource(testPath);
        Instances testInstances = loader.getDataSet();
        testInstances.setClassIndex(testInstances.numAttributes()-1);
        Evaluation eval = new Evaluation(testInstances);
        eval.evaluateModel(ruleMoldel,testInstances);
        ArrayList<Prediction> predictions = eval.predictions();
        int predictedVal = (int)predictions.get(0).predicted();
        String cdetails = instances.classAttribute().value(predictedVal);
        return cdetails;
    }

    private void loadSaveModel(String filePath,boolean status){

        if(status){

            try {

                ObjectOutputStream outStream = new ObjectOutputStream(new FileOutputStream(filePath));
                outStream.writeObject(ruleMoldel);
                outStream.flush();
                outStream.close();
            } catch (IOException e) {

                log.error("Error occurred:"+e.getMessage());
            }
        }else{

            try {

                ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(filePath));
                ruleMoldel = (JRip) inputStream.readObject();
            } catch (IOException e) {

                log.error("Error occurred:"+e.getMessage());
            } catch (ClassNotFoundException e) {

                log.error("Error occurred:"+e.getMessage());
            }

        }
    }
}
