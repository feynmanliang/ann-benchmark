import org.apache.log4j._
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

Logger.getRootLogger.setLevel(Level.OFF)

// maximum number of worker nodes in cluster
val numNodes= 5

// number of iterations to run
val numIterations = 5

val trainData = MLUtils.loadLibSVMFile(sc, "data/mllib/mnist.scale")
val layers = Array[Int](780, 2500, 2000, 1500, 1000, 500, 10)
val estimator = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setMaxIter(numIterations)
  .setSeed(11L)

// parallalize the data for N nodes, persist, run X iterations and print average time for each run
for (i <- 1 to numNodes) {
  val dataPartitions = sc.parallelize(1 to i, i)
  val sample = trainData.sample(true, 1.0 / i, 11L).collect
  val parallelData = dataPartitions.flatMap(x => sample).toDF()

  // cache and materialize dataset
  parallelData.cache()
  parallelData.count()

  // run MLP training and print results
  val t = System.nanoTime()
  val model = estimator.fit(parallelData)
  println(i + "\t" + (System.nanoTime() - t) / (numIterations * 1e9))
}
