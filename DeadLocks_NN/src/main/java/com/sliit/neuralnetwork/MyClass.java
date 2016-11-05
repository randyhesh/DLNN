package com.sliit.neuralnetwork;

import org.apache.commons.io.FileUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;


/**
 * Created by Heshani on 11/1/2016.
 */
public class MyClass {

    public static void main(String[] args) {



    try {
        File newF=new File("D:/Data/aa");

        if (newF.exists()) {
            FileUtils.deleteDirectory(newF);
        }


        BufferedReader br = new BufferedReader(new FileReader("D:/Data/a.csv"));
        String line = br.readLine();


        while (line!=null){
            System.out.println(line);
            line = br.readLine();

        }


    }catch (Exception e){
        e.printStackTrace();
    }

    }
}
