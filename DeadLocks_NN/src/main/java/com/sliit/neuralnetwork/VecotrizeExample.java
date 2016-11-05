package com.sliit.neuralnetwork;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.canova.api.conf.Configuration;
import org.canova.api.vector.Vectorizer;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.regex.RegexLineRecordReader;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.LongColumnCondition;
import org.datavec.api.transform.condition.string.StringRegexColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.quality.DataQualityAnalysis;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.categorical.CategoricalToIntegerTransform;
import org.datavec.api.transform.transform.doubletransform.DoubleMathOpTransform;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.AnalyzeSpark;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.joda.time.DateTimeZone;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.util.List;
import java.util.zip.GZIPInputStream;
/**
 * Created by heshani on 7/29/16.
 */
public class VecotrizeExample {

    public static final String DATA_URL = "ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz";
    /** Location to save and extract the training/testing data */
    public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("user.dir"), "datavec_log_example/");
    public static final String EXTRACTED_PATH = FilenameUtils.concat(DATA_PATH, "data/data.txt");

    public static void main(String[] args) throws Exception {
        //Setup
        //downloadData();
        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DataVec Log Data Example");
        JavaSparkContext sc = new JavaSparkContext(conf);


        //=====================================================================
        //                 Step 1: Define the input data schema
        //=====================================================================

        //First: let's specify a schema for the data. This is based on the information from: http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html
        Schema schema = new Schema.Builder()
                .addColumnInteger("userID")
                .addColumnDouble("sourceIp")
                .addColumnDouble("destinationIp")
                .addColumnDouble("timestampdiff")
                .addColumnCategorical("status","normal","fraud")
                .build();

        //=====================================================================
        //                     Step 2: Clean Invalid Lines
        //=====================================================================

        //Second: let's load the data. Initially as Strings
        JavaRDD<String> logLines = sc.textFile(EXTRACTED_PATH);

        //This data unfortunately contains a small number of invalid lines. We'll remove them using standard Spark functionality

        //=====================================================================
        //         Step 3: Parse Raw Data and Perform Initial Analysis
        //=====================================================================


        RecordReader rr = new CSVRecordReader(0,",");
        JavaRDD<List<Writable>> parsed = logLines.map(new StringToWritablesFunction(rr));

        //Now, let's check the quality, so we know if there's anything we need to clean up first...
        //DataQualityAnalysis dqa = AnalyzeSpark.analyzeQuality(schema, parsed);
        //System.out.println("----- Data Quality -----");
        //System.out.println(dqa);    //One issue: non-integer values in "replyBytes" column


        //=====================================================================
        //          Step 4: Perform Cleaning, Parsing and Aggregation
        //=====================================================================

        //Let's specify the transforms we want to do
        TransformProcess tp = new TransformProcess.Builder(schema)
                //First: clean up the "replyBytes" column by replacing any non-integer entries with the value 0
                //Second: let's parse the date/time string:
                .build();
        SparkTransformExecutor executor = new SparkTransformExecutor();
        JavaRDD<List<Writable>> processed = executor.execute(parsed, tp);
        processed.cache();


        //=====================================================================
        //       Step 5: Perform Analysis on Final Data; Display Results
        //=====================================================================

        Schema finalDataSchema = tp.getFinalSchema();
        long finalDataCount = processed.count();
        List<List<Writable>> sample = processed.take(10);

        //DataAnalysis analysis = AnalyzeSpark.analyze(finalDataSchema, processed);

        sc.stop();
        Thread.sleep(4000); //Give spark some time to shut down (and stop spamming console)


        System.out.println("----- Final Data Schema -----");
        System.out.println(finalDataSchema);

        System.out.println("\n\nFinal data count: " + finalDataCount);

        System.out.println("\n\n----- Samples of final data -----");
        for(List<Writable> l : sample){
            System.out.println(l);
        }

        //System.out.println("\n\n----- Analysis -----");
        //System.out.println(analysis);
    }


    private static void downloadData() throws Exception {
        //Create directory if required
        File directory = new File(DATA_PATH);
        if(!directory.exists()) directory.mkdir();

        //Download file:
        String archivePath = DATA_PATH + "NASA_access_log_Jul95.gz";
        File archiveFile = new File(archivePath);
        File extractedFile = new File(EXTRACTED_PATH,"access_log_July95.txt");
        new File(extractedFile.getParent()).mkdirs();

        if( !archiveFile.exists() ){
            System.out.println("Starting data download (20MB)...");
            FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile);
            System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
            //Extract tar.gz file to output directory
            extractGzip(archivePath, extractedFile.getAbsolutePath());
        } else {
            //Assume if archive (.tar.gz) exists, then data has already been extracted
            System.out.println("Data (.gz file) already exists at " + archiveFile.getAbsolutePath());
            if( !extractedFile.exists()){
                //Extract tar.gz file to output directory
                extractGzip(archivePath, extractedFile.getAbsolutePath());
            } else {
                System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
            }
        }
    }

    private static final int BUFFER_SIZE = 4096;
    private static void extractGzip(String filePath, String outputPath) throws IOException {
        System.out.println("Extracting files...");
        byte[] buffer = new byte[BUFFER_SIZE];

        try{
            GZIPInputStream gzis = new GZIPInputStream(new FileInputStream(new File(filePath)));

            FileOutputStream out = new FileOutputStream(new File(outputPath));

            int len;
            while ((len = gzis.read(buffer)) > 0) {
                out.write(buffer, 0, len);
            }

            gzis.close();
            out.close();

            System.out.println("Done");
        }catch(IOException ex){
            ex.printStackTrace();
        }
    }

}
