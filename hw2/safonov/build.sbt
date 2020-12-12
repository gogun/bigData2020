version := "0.1"

scalaVersion := "2.10.7"

dependencyOverrides += "org.scala-lang" % "scala-compiler" % scalaVersion.value

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.2.3"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.2.3"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.3"
libraryDependencies += "com.github.master" %% "spark-stemming" % "0.2.1"
