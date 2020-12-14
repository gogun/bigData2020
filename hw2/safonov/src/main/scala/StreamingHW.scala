import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, from_json}
import org.apache.spark.sql.types.{IntegerType, StringType, StructType}

object StreamingHW {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("hw3")
      .config("spark.master", "local")
      .getOrCreate()

    spark.sparkContext.setLogLevel("OFF")

    val schema = new StructType()
      .add("id", StringType, nullable = true)
      .add("text", StringType, nullable = true)

    val inputData = spark.readStream
      .format("socket")
      .option("host", "localhost")
      .option("port", 8065)
      .load()

    val inputJson =
      inputData.withColumn("json", from_json(col("value"), schema))
        .select("json.*")
        .select(col("id"), col("text"))

    val model = PipelineModel.read.load("model/")
    inputJson.printSchema()
    model.transform(inputJson.select(col("id"), col("text")))
      .select(col("id"), col("target").as("target").cast(IntegerType))
        .repartition(1)
          .writeStream
            .outputMode("append")
              .format("com.databricks.spark.csv")
                .option("header", "true")
                .option("path", "src/main/resources/path/")
                .option("checkpointLocation", "checkpointLocation/")
                  .start()
                    .awaitTermination()
  }
}
