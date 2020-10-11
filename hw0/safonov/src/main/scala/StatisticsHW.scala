import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{asc, callUDF, col, desc, lit, row_number, stddev}

object StatisticsHW {
  def main(args: Array[String]) {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("hw0")
      .config("spark.master", "local")
      .getOrCreate()

    val data = spark.read
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .option("escape", "\"")
      .csv("/home/mrsandman5/bigData2020/hw0/safonov/src/main/resources/AB_NYC_2019.csv")

    val settings = data
      .withColumn("latitude", data("latitude").cast("Double"))
      .withColumn("longitude", data("longitude").cast("Double"))
      .withColumn("price", data("price").cast("Integer"))
      .withColumn("minimum_nights", data("minimum_nights").cast("Integer"))
      .withColumn("number_of_reviews", data("number_of_reviews").cast("Integer"))
      .where(col("price") > 0)
      .na
      .drop()

    //Посчитать медиану для каждого room_type
    println("Median")
    settings
      .groupBy("room_type")
      .agg(
        callUDF("percentile_approx", col("price"), lit(0.5)).as("median")
      )
      .show()

    //Посчитать медиану для каждого room_type
    println("Mode")
    val room_settings = settings
      .groupBy("room_type", "price")
      .count()
    val window = Window.partitionBy("room_type").orderBy(desc("count"))
    room_settings.withColumn("row_number", row_number().over(window))
      .select("room_type", "price")
      .where(col("row_number") === 1)
      .show()

    //Посчитать среднее для каждого room_type
    println("Mean")
    settings
      .groupBy("room_type")
      .mean("price")
      .show()

    //Посчитать дисперсию для каждого room_type
    println("Dispersion")
    settings.select("room_type", "price")
      .groupBy("room_type")
      .agg(stddev("price"))
      .show()

    //Найти самое дорогое и самое дешевое предложение
    println("Max price")
    settings.orderBy(desc("price")).show(1)
    println("Min price")
    settings.orderBy(asc("price")).show(1)

    //Посчитать корреляцию между ценой и минимальный количеством ночей, кол-вом отзывов
    println("Correlation between price and minimum_nights")
    println(settings.stat.corr("price", "minimum_nights", "pearson"))
    println("Correlation between price and number_of_reviews")
    println(settings.stat.corr("price", "number_of_reviews", "pearson"))

    //Найти гео квадрат размером 5км на 5км с самой высокой средней стоимостью жилья


  }
}
