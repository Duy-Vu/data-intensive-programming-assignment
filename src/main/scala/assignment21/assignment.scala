package assignment21

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.{window, column, desc, col}


import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Column
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, IntegerType, DoubleType}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.{count, sum, min, max, asc, desc, udf, to_date, avg}

import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.SparkSession

import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}




import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansSummary}


import java.io.{PrintWriter, File}


//import java.lang.Thread
import sys.process._


import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection.immutable.Range

// CHECK LIST
//  TODO TASK 1   DONE
//  TODO TASK 2   DONE
//  TODO TASK 3   DONE
//  TODO TASK 4   DONE
//  TODO BONUS #1 DONE
//  TODO BONUS #2 DONE
//  TODO BONUS #3 DONE 
//  TODO BONUS #4 DONE
//  TODO BONUS #5 NOT DONE
//  TODO BONUS #6 DONE

object assignment {
  // Suppress the log messages:
  Logger.getLogger("org").setLevel(Level.OFF)
                       
  
  val spark = SparkSession.builder()
    .appName("assignment")
    .config("spark.driver.host", "localhost")
    .master("local")
    .getOrCreate()

  spark.conf.set("spark.sql.shuffle.partitions", "5")

  // Create schema for data validation if need ----------
    val K5D2Schema = new StructType(Array(
    new StructField("a", DoubleType, false),
    new StructField("b", DoubleType, false),
    new StructField("LABEL", StringType, false)))
  val K5D3Schema = new StructType(Array(
    new StructField("a", DoubleType, false),
    new StructField("b", DoubleType, false),
    new StructField("c", DoubleType, false),
    new StructField("LABEL", StringType, false)))
  // An alternative solution for na.drop() is that:
  // With the schema, we can already get a runtime error during the reading execution if there is an invalid element in the read data.
  
  // Read data with care of Null value --------------------BONUS TASK #3
  val dataK5D2 = spark.read
    .options(
      Map("delimiter" -> ",","format" -> "csv","inferSchema"-> "true","header"->"true"))
    .csv("./data/dataK5D2.csv")
    .na.drop("any")
  val dataK5D3 = spark.read
    .options(
      Map(
        "delimiter" -> ",",
        "format" -> "csv"
        ,"inferSchema"-> "true","header"->"true"))
    .csv("./data/dataK5D3.csv")
    .na.drop("any")
    

  dataK5D2.show()
  dataK5D3.show()
  
  dataK5D2.printSchema()
  dataK5D3.printSchema()
  
  // // example checking if 1 column has null
//  if(dataK5D2.col("a").isNull == true) println("a has null")
//  else println("a is has no null")
  
  
  // ---------------- TASK 1 --------------------- 
  import org.apache.spark.mllib.linalg.Vectors
  import org.apache.spark.ml.feature.MinMaxScaler

  // Function returns min and max value of a column in an array
  // getMinMax is only needed for features columns
  def getMinMax(df: DataFrame): Array[Double] = {
    val columnNames = df.drop("LABEL").columns.toSeq
    val aggValue = columnNames.map(colName => df.agg(min(colName),max(colName)).head().toSeq.toArray)
    val minMax = aggValue.flatten.map(_.toString.toDouble).toArray
    return minMax
  }
  
  // ------------- BONUS TASK #6 ------------ 
  // Re-scale centers function
  def getRescaleCenter2D(centers: Array[(Double, Double)], minMaxArray : Array[Double]) : Array[(Double, Double)] = {
    val aMin = minMaxArray(0)
    val aMax = minMaxArray(1)
    val bMin = minMaxArray(2)
    val bMax = minMaxArray(3)

    
    return centers.map(center => (center._1*(aMax-aMin)+aMin,center._2*(bMax-bMin)+bMin))
  }
 
  
  def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    val vectorAssembler = new VectorAssembler()
                              .setInputCols(Array("a","b"))
                              .setOutputCol("features")
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures").setMin(0).setMax(1)
  
    
    val assembledDF = vectorAssembler.transform(df)
    val scaledDF = scaler.fit(assembledDF).transform(assembledDF)
    
//    scaledDF.show(10)
    val kMeansNotScaled = new KMeans()
                     .setK(k)
                     .setSeed(1L)
    
    val kMeansModelNotScaled = kMeansNotScaled.fit(assembledDF)
    val clusterCentersNotScaled = kMeansModelNotScaled.clusterCenters
//    println("not scaled result:")
//    clusterCentersNotScaled.foreach(println)
    
    
    val kMeans = new KMeans()
                     .setK(k)
                     .setMaxIter(200)
                     .setSeed(1L)
                     .setFeaturesCol("scaledFeatures")
//    println(kMeans.explainParams)
    val kMeansModel = kMeans.fit(scaledDF)
    
//    println("prediction show")
//    kMeansModel.summary.predictions.show
//    kMeansModel.summary.predictions.select("prediction").distinct().show
    val clusterCenters = kMeansModel.clusterCenters

//    println("scaled result:")
//    clusterCenters.foreach(println)
    val scaledCenters = clusterCenters.map(center => (center(0), center(1)))
    val minMaxArray = getMinMax(df)
    val scaledBackCenters = getRescaleCenter2D(scaledCenters,minMaxArray)

//    println("rescaled result:")
//    scaledBackCenters.foreach(println)
    return scaledBackCenters
  }

  //-------------Task 2------------- 
  def getRescaleCenter3D(centers: Array[(Double, Double, Double)], minMaxArray : Array[Double]) : Array[(Double, Double, Double)] = {
    val aMin = minMaxArray(0)
    val aMax = minMaxArray(1)
    val bMin = minMaxArray(2)
    val bMax = minMaxArray(3)
    val cMin = minMaxArray(4)
    val cMax = minMaxArray(5)
    
    
    return centers.map(center => (center._1*(aMax-aMin)+aMin,center._2*(bMax-bMin)+bMin,center._3*(cMax-cMin)+cMin))
  }  
  
  def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
   val vectorAssembler = new VectorAssembler()
                              .setInputCols(Array("a","b","c"))
                              .setOutputCol("features")
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures").setMin(0).setMax(1)
    val assembledDF = vectorAssembler.transform(df)
    
    val scaledDF = scaler.fit(assembledDF).transform(assembledDF)
    
//    scaledDF.show(10)
    val kMeansNotScaled = new KMeans()
                     .setK(k)
                     .setSeed(1L)
    
    val kMeansModelNotScaled = kMeansNotScaled.fit(assembledDF)
    val clusterCentersNotScaled = kMeansModelNotScaled.clusterCenters
//    println("not scaled result:")
//    clusterCentersNotScaled.foreach(println)
    
    
    val kMeans = new KMeans()
                     .setK(k)
                     .setSeed(1L)
                     .setFeaturesCol("scaledFeatures")
    val kMeansModel = kMeans.fit(scaledDF)
    val clusterCenters = kMeansModel.clusterCenters
//    println("scaled result:")
//    clusterCenters.foreach(println)
    val scaledCenters = clusterCenters.map(center => (center(0), center(1),center(2)))
    val minMaxArray = getMinMax(df)
    val scaledBackCenters = getRescaleCenter3D(scaledCenters,minMaxArray)
    
//    println("rescaled result:")
//    scaledBackCenters.foreach(println)
    return scaledBackCenters 
  }
  
  //----------------- TASK 3 ---------------- 
  
  import org.apache.spark.sql.functions.{trim, lower, when}
  // use trim and lower to get the acceptable data out of the original data, transform any other into lable 2
  // e.g: we can accept the space and misplaced capital letters
  //
  val dataK5D2WithLabels : DataFrame = dataK5D2.withColumn("num(LABEL)",
                                     when(trim(lower(col("LABEL"))) === "ok",0)
                                     .when(trim(lower(col("LABEL"))) === "fatal",1)
                                     .otherwise(2))
                                     
  val dataK5D3WithLabels : DataFrame = dataK5D3.withColumn("num(LABEL)",
                                     when(trim(lower(col("LABEL"))) === "ok",0)
                                     .when(trim(lower(col("LABEL"))) === "fatal",1)
                                     .otherwise(2))

                                     
 // NOTE In task 3, use only 2 columns "a" and "b" even there are 3, confirmed by course instructor
                                     
  def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
    val cleanedData = df.filter(col("num(LABEL)") !== 2)
    
    val vectorAssembler = new VectorAssembler()
                              .setInputCols(Array("a","b", "num(LABEL)"))
                              .setOutputCol("features")
    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures").setMin(0).setMax(1)
    val assembledDF = vectorAssembler.transform(cleanedData)
    println("Assembled DF: ")
    assembledDF.limit(10).show(false)
    
    val scaledDF = scaler.fit(assembledDF).transform(assembledDF)
    scaledDF.limit(10).show(false)
       
    val kMeans = new KMeans()
                     .setK(k)
                     .setSeed(1L)
                     .setFeaturesCol("scaledFeatures")
    val kMeansModel = kMeans.fit(scaledDF)
    val clusterCenters = kMeansModel.clusterCenters
//    println("All clusters: ")
//    clusterCenters.foreach(println)

    val scaledCenters = clusterCenters.map(center => (center(0), center(1), center(2)))
    println("In scaled 0-1 centers: ")
    scaledCenters.foreach(println)
    
    // NOTE: As we don't need "c" column, remove its min and max from our array
    val minMaxArray = getMinMax(df)
//    println("MinMax values init")
//    minMaxArray.foreach(println)
    val minMaxArrayNew = minMaxArray.slice(0,4) ++ minMaxArray.slice(6,8)
//    println("MinMax values")
//    minMaxArrayNew.foreach(println)
    
    val scaledBackCenters = getRescaleCenter3D(scaledCenters,minMaxArrayNew)
//    println("Scaled back centers")
//    scaledBackCenters.foreach(println)
    
    // Sort the centers predicted values num(LABEL) and get the closest-to-1 values 
    val selectedMeans = scaledBackCenters.sortWith(_._3 > _._3).take(2)
    println("Highest fatal")
    selectedMeans.foreach(println)
    return selectedMeans.map(mean => (mean._1, mean._2))
  }

  // ------------------------ TASK 4 -------------------- 
  
  // Parameter low is the lowest k and high is the highest one.
  def costCalculate(df: DataFrame, k: Int):(Int, Double) = {
    val vectorAssembler = new VectorAssembler()
                              .setInputCols(Array("a","b"))
                              .setOutputCol("features")
    val assembledDF = vectorAssembler.transform(df)
    val KMeans = new KMeans()
                          .setK(k)
                          .setSeed(1L)
    val KMeansModel = KMeans.fit(assembledDF)
    
    return (k,KMeansModel.computeCost(assembledDF))
  }
  
  import breeze.linalg._
  import breeze.plot._
  import breeze.numerics._
 
  
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {
    val kValues : Array[Int] = (low to high).toArray
    val costByK = kValues.map(k => costCalculate(df,k))
    
    // Visualization
    val fig = Figure()
    val p = fig.subplot(0)
    val cost = new DenseVector(costByK.map(_._2))
    val clusterAmount = new DenseVector(kValues.map(x => x.toDouble))
    p += plot(clusterAmount, cost)
    
    return costByK
  }
     
}




