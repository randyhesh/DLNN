package com.sliit.neuralnetwork;

import org.canova.api.records.reader.RecordReader;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.spark.canova.RecordReaderFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by heshani on 7/4/16.
 */
public class SparkDataSet {

    private SparkConf conf;
    private JavaSparkContext sc = null;
    private RecordReader recordReader;
    private int labelIndex;
    private int numOutPutClasses;
    private static SparkDataSet sparkDataSet = null;

    private SparkDataSet(String sparkAppName,String master,int labels,int output){

        conf = new SparkConf().setAppName(sparkAppName).setMaster(master);
        setSc(new JavaSparkContext(conf));
        recordReader = new CSVRecordReader(0,",");
        labelIndex = labels;
        numOutPutClasses = output;
    }

    public JavaRDD<DataSet> generateTrainingDataset(String hdfs_url) {

        JavaRDD<String> distFile = getSc().textFile(hdfs_url);
        JavaRDD<DataSet> trainingData = distFile.map(new RecordReaderFunction(recordReader, labelIndex, numOutPutClasses));
        return trainingData;
    }


    public static SparkDataSet getInstance(String sparkAppName,String master,int labels,int outputs){

        if(sparkDataSet == null){

            sparkDataSet = new SparkDataSet(sparkAppName,master,labels,outputs);
        }
        return sparkDataSet;
    }

    public JavaSparkContext getSc() {
        return sc;
    }

    public void setSc(JavaSparkContext sc) {
        this.sc = sc;
    }
}
