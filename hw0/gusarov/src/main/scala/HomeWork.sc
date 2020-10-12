import ch.hsr.geohash.GeoHash
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType}

val spark = SparkSession
  .builder()
  .appName("hw0")
  .config("spark.master", "local")
  .getOrCreate()

var df = spark.read.options(Map("header" -> "true", "escape" -> "\"", "multiLine" -> "true"))
  .csv("Документы/opensource_coding/bigData2020/hw0/gusarov/data/AB_NYC_2019.csv")
df = df.withColumn("price", col("price").cast(IntegerType))

df.groupBy("room_type").mean("price").show()

df.groupBy("room_type", "price")
  .count().sort(desc("count")).groupBy("room_type")
  .agg(first("price").alias("mode"))
  .show()

df.groupBy("room_type")
  .agg(expr("percentile_approx(price, 0.5)").alias("med"))
  .show()

df.groupBy("room_type")
  .agg((stddev("price") * stddev("price")).alias("dispersion"))
  .show()
print("MOST EXPENSIVE OFFER")
df.sort(desc("price")).show(1)
print("CHEAPEST OFFER")
df.sort("price").show(1)

df.withColumn("minimum_nights", col("minimum_nights").cast(IntegerType))
  .stat.corr("price", "minimum_nights")

df.withColumn("number_of_reviews", col("number_of_reviews").cast(IntegerType))
  .stat.corr("price", "number_of_reviews")

val geohash_udf = udf(GeoHash.geoHashStringWithCharacterPrecision _)
df.withColumn("latitude", col("latitude").cast(DoubleType))
  .withColumn("longitude", col("longitude").cast(DoubleType))
  .withColumn("geohash", geohash_udf(col("latitude"), col("longitude"), lit(5)).alias("geohash"))
  .groupBy("geohash")
  .mean("price")
  .sort(desc("avg(price)"))
  .show(1)
