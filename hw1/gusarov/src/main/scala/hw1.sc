import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{RegexTokenizer, StringIndexer, Word2Vec}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{column, lit, when}
import org.apache.spark.sql.types.{IntegerType, LongType}

val spark = SparkSession
  .builder()
  .appName("hw1")
  .config("spark.master", "local")
  .getOrCreate()

var df = spark.read.options(Map("header" -> "true", "escape" -> "\""))
  .csv("Documents/coding/bigData2020/hw1/gusarov/data/train.csv")

df = df.na.drop()
df = df
  .withColumn("id", df("id").cast(LongType))
  .withColumn("target", df("target").cast(IntegerType))

df.show()
//pipeline
val regexTokenizer = new RegexTokenizer()
  .setInputCol("text")
  .setOutputCol("words")
  .setPattern("[\\W]")

val stemmer = new Stemmer()
  .setInputCol("words")
  .setOutputCol("stemmed")

val word2Vec = new Word2Vec()
  .setInputCol("stemmed")
  .setOutputCol("features")

val stringIndexer = new StringIndexer()
  .setInputCol("target")
  .setOutputCol("label")

val gbt = new GBTClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setPredictionCol("prediction")
  .setMaxIter(25)

val pipeline = new Pipeline()
  .setStages(Array(regexTokenizer, stemmer, stringIndexer, word2Vec, gbt))

//cv
val paramGrid = new ParamGridBuilder()
  .addGrid(gbt.maxDepth, Array(3, 30))
  .addGrid(word2Vec.vectorSize, Array(90, 110))
  .build()

val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(2)
  .setParallelism(4)

val bestEstimator = cv.fit(df)

//test
var testDf = spark.read
  .options(Map("header" -> "true", "escape" -> "\""))
  .csv("Documents/coding/bigData2020/hw1/gusarov/data/test.csv")
  .filter(column("id").isNotNull)
  .filter(column("text").isNotNull)

print(testDf.count())
testDf = testDf
  .withColumn("id", testDf("id").cast(LongType))
  .select("id", "text")
testDf.show()
val predictedData = bestEstimator.transform(testDf)
predictedData.show()
val result = predictedData
  .withColumn("target", column("prediction").cast(IntegerType))
  .select("id", "target")
result.show()


val sample = spark.read
  .options(Map("header" -> "true", "escape" -> "\""))
  .csv("Documents/coding/bigData2020/hw1/gusarov/data/sample_submission.csv")
  .withColumn("id", column("id").cast(LongType))
  .select("id")

val output = sample.join(result, sample("id").equalTo(result("id")), "left")
  .select(sample("id"), when(result("id").isNull, lit(0)).otherwise(column("target")).as("target"))

output.write
  .format("com.databricks.spark.csv")
  .option("header", "true")
  .save("Documents/coding/bigData2020/hw1/gusarov/data/prediction1.csv")
