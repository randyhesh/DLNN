package com.sliit.neuralnetwork;

import java.io.*;

/**
 * Created by heshani on 8/12/16.
 */
public class IPtoDecimalConverter {

    public static void main(String[] args){

        IPtoDecimalConverter ip = new IPtoDecimalConverter();
        //ip.convertIP("fraud_kdd.csv");
        ip.convertFormat("normalizedkddweka.txt");
    }

    public long ipToLong(String ipAddress) {

        String[] ipAddressInArray = ipAddress.split("\\.");

        long result = 0;
        for (int i = 0; i < ipAddressInArray.length; i++) {

            int power = 3 - i;
            int ip = Integer.parseInt(ipAddressInArray[i]);
            result += ip * Math.pow(256, power);

        }

        return result;
    }

    public void convertIP(String path){

        File file = new File(path);
        try {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            File newFile = new File("normalizedkdd.txt");
            if(!newFile.exists()){

                newFile.createNewFile();
            }
            BufferedReader write = new BufferedReader(new FileReader(new File("normalizedkdd.txt")));
            PrintWriter printWriter = new PrintWriter(newFile);
            String line = "";
            String outputWrite = "";
            int count = 0;
            while((line = reader.readLine())!=null){

                if(count==0){

                    count++;
                    continue;
                }
                outputWrite = "";
                String values[] = line.split(",");
                int i=0;
                for(String value:values){

                    if(i == 4||i==5){

                        value = ipToLong(value)+"";
                        outputWrite = outputWrite.concat(","+value);
                    }
                    else{

                        if(outputWrite.equals("")){

                            outputWrite = value;
                        }else {
                            outputWrite = outputWrite.concat("," + value);
                        }
                    }
                    i++;
                }
                printWriter.println(outputWrite);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void convertFormat(String filePath){

        try {
            BufferedReader reader = new BufferedReader(new FileReader(filePath));
            String outputWrite = "";
            int count = 0;
            String line="";
            File newFile = new File("normalizedkddwekaformat.txt");
            if(!newFile.exists()){

                newFile.createNewFile();
            }
            BufferedReader write = new BufferedReader(new FileReader(new File("normalizedkdd.txt")));
            PrintWriter printWriter = new PrintWriter(newFile);
            while((line = reader.readLine())!=null) {

                String[] values = line.split(",");
                for (String value : values) {

                    if (count == 0) {

                        outputWrite = value;
                    } else {
                        outputWrite = outputWrite.concat("," + value);
                    }
                    if (count % 100 == 0 && count != 0) {

                        printWriter.println(outputWrite);
                        outputWrite="";
                        count = 0;
                        continue;
                    }
                    count++;
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public void lineReader(String filePath){

        try {
            BufferedReader reader = new BufferedReader(new FileReader(filePath));
            String outputWrite = "";
            int count = 0;
            String line="";
            File newFile = new File("normalizedkddwekaline.txt");
            if(!newFile.exists()){

                newFile.createNewFile();
            }
            BufferedReader write = new BufferedReader(new FileReader(new File("normalizedkdd.txt")));
            PrintWriter printWriter = new PrintWriter(newFile);
            while((line = reader.readLine())!=null) {

                String[] values = line.split(",");
                for (String value : values) {

                    if(count==0){

                        outputWrite = value;
                    }
                    else{
                        outputWrite=outputWrite.concat(","+value);
                    }
                    count++;
                }
                printWriter.println(outputWrite);
                outputWrite="";
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
