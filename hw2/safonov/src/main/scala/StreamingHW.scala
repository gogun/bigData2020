import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, StringIndexer}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, lit, when}
import org.apache.spark.sql.types.DataTypes

object StreamingHW {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("hw2")
      .config("spark.master", "local")
      .getOrCreate()

    val train = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("/home/mrsandman5/bigData2020/hw1/safonov/src/main/resources/train.csv")
      .filter(col("id").isNotNull)
      .filter(col("target").isNotNull)
      .filter(col("text").isNotNull)
      .select("id", "text", "target")
      .withColumnRenamed("target", "label")

    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("words")
      .setPattern("[\\W]")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("removed")

    val stemmer = new Stemmer()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("stemmed")
      .setLanguage("English")

    val hashingTF = new HashingTF()
      .setInputCol(stemmer.getOutputCol)
      .setNumFeatures(5000)
      .setOutputCol("rowFeatures")

    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("features")

    val stringIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")

    val gbt = new GBTClassifier()
      .setLabelCol(stringIndexer.getOutputCol)
      .setFeaturesCol(idf.getOutputCol)
      .setPredictionCol("target")
      .setMaxIter(30)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, stemmer, hashingTF, idf, stringIndexer, gbt))

    val model = pipeline.fit(train)

    val test = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("/home/mrsandman5/bigData2020/hw1/safonov/src/main/resources/test.csv")
      .filter(col("id").isNotNull)
      .filter(col("text").isNotNull)
      .select("id", "text")

    val predictions = model.transform(test)
    val result = predictions.select(col("id"), col("target").cast(DataTypes.IntegerType))

    val sample = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("/home/mrsandman5/bigData2020/hw1/safonov/src/main/resources/sample_submission.csv")
      .select("id")

    result.join(sample, sample.col("id").equalTo(result.col("id")), "right")
      .select(sample.col("id"), when(result.col("id").isNull, lit(0))
        .otherwise(col("target"))
        .as("target"))
      .write.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .save("/home/mrsandman5/bigData2020/hw1/safonov/src/main/resources/my_result.csv")

  }
}
