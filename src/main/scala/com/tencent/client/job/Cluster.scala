package com.tencent.client.job
import java.io.{File, PrintWriter}
import java.nio.file.{Files, Paths}
import java.net._
import java.util.UUID

import com.tencent.client.worker.Worker
import com.tencent.client.master.Master
import org.apache.spark.deploy.SparkHadoopUtil
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext, SparkFiles}
import com.tencent.client.common.protos.AsyncModel
import com.tencent.client.plasma.PlasmaClient
import com.tencent.angel.conf.AngelConf._
import org.apache.hadoop.conf.Configuration
import com.tencent.angel.ps.ParameterServer
import com.tencent.angel.conf.AngelConf
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkConf
import org.apache.spark.deploy.SparkHadoopUtil

import scala.collection.mutable.ArrayBuffer

object Cluster {
  def addUserResourceFiles(sparkConf: SparkConf, conf: Configuration): Unit = {
    val appStagingBaseDir = sparkConf.getOption("spark.yarn.stagingDir")
      .fold(FileSystem.get(conf).getHomeDirectory)(new Path(_))
    val appStagingDir = new Path(appStagingBaseDir, s".sparkStaging/${sparkConf.getAppId}")
    val resourceFiles = new ArrayBuffer[String]()
    sparkConf.getOption("spark.yarn.dist.files").foreach { fileList =>
      fileList.split(",").foreach { file =>
        val fileName = file.trim.split("/").last
        if (fileName.nonEmpty) {
          resourceFiles.append(new Path(appStagingDir, fileName).toString)
        }
      }
      conf.set(AngelConf.ANGEL_APP_USER_RESOURCE_FILES, resourceFiles.mkString(","))
    }
  }
  def convertToHadoop(conf: SparkConf): Configuration = {
    val MASTER_IP = "angel.master.ip"
    val MASTER_PORT = "angel.master.port"
    val TOTAL_CORES = "angel.ps.total.cores"
    val CONF_KEYS = "angel.conf.keys"
    val ALGO_TYPE = "angel.algo.type"
    val appName = conf.get("spark.app.name") + "-ps"
    val queue = conf.get("spark.yarn.queue", "root.default")

    /** mode: YARN , KUBERNETES or LOCAL */
    val master = conf.getOption("spark.master")
    val isLocal = if (master.isEmpty || master.get.toLowerCase.startsWith("local")) true else false
    val deployMode = if (isLocal) "LOCAL" else conf.get("spark.ps.mode", DEFAULT_ANGEL_DEPLOY_MODE)

    val masterMem = conf.getSizeAsGb("spark.angel.master.memory", "2g").toInt
    val psNum = conf.getInt("spark.ps.instances", 1)
    val psCores = conf.getInt("spark.ps.cores", 1)
    val psMem = conf.getSizeAsGb("spark.ps.memory", "4g").toInt
    val psOpts = conf.getOption("spark.ps.extraJavaOptions")

    val psJars = conf.get("spark.ps.jars", "")
    val psLogLevel = conf.get("spark.ps.log.level", "INFO")
    val psClass = conf.get("spark.ps.class", classOf[ParameterServer].getName)

    val defaultFS = conf.get("spark.hadoop.fs.defaultFS", "file://")
    val tempPath = defaultFS + "/tmp/spark-on-angel/" + UUID.randomUUID()

    val psOutOverwrite = conf.getBoolean("spark.ps.out.overwrite", defaultValue = true)
    val psOutTmpOption = conf.getOption("spark.ps.out.tmp.path.prefix")


    val hadoopConf = SparkHadoopUtil.get.newConfiguration(conf)
    val filterExp = List("spark.ps", "spark.hadoop", "angel", "ml")
    conf.getAll.foreach{
      case (key: String, value: String) if filterExp.exists(expr => key.startsWith(expr)) =>
        hadoopConf.set(key, value)
      case _ =>
    }

    // setting running mode, app name, queue and deploy mode
    hadoopConf.set(ANGEL_RUNNING_MODE, "ANGEL_PS")
    hadoopConf.set(ANGEL_JOB_NAME, appName)
    hadoopConf.set(ANGEL_QUEUE, queue)
    hadoopConf.set(ANGEL_DEPLOY_MODE, deployMode)

    // For local mode, we set heartbeat a small value for fast debugging
    if (deployMode == "LOCAL")
      hadoopConf.set(ANGEL_PS_HEARTBEAT_INTERVAL_MS, "200")

    if (psOpts.isDefined)
      hadoopConf.set(ANGEL_PS_JAVA_OPTS, psOpts.get)

    // Set the temp path as the angel.save.model.path to fake the angel-ps system
    // The action type is also a fake setting.
    hadoopConf.set(ANGEL_ACTION_TYPE, "train")
    hadoopConf.set(ANGEL_SAVE_MODEL_PATH, tempPath)

    // Setting resource
    hadoopConf.setInt(ANGEL_AM_MEMORY_GB, masterMem)

    hadoopConf.setInt(ANGEL_PS_NUMBER, psNum)
    hadoopConf.setInt(ANGEL_PS_CPU_VCORES, psCores)
    hadoopConf.setInt(ANGEL_PS_MEMORY_GB, psMem)
    hadoopConf.setInt(TOTAL_CORES, psNum * psCores)

    hadoopConf.set(ANGEL_AM_LOG_LEVEL, psLogLevel)
    hadoopConf.set(ANGEL_PS_LOG_LEVEL, psLogLevel)
    hadoopConf.set(ANGEL_PS_CLASS, psClass)
    hadoopConf.set(ANGEL_JOB_LIBJARS, psJars)

    hadoopConf.setBoolean(ANGEL_JOB_OUTPUT_PATH_DELETEONEXIST, psOutOverwrite)
    psOutTmpOption.foreach(hadoopConf.set(ANGEL_JOB_TMP_OUTPUT_PATH_PREFIX, _))

    // No need for sync clock values, we don't need ssp in Spark.
    hadoopConf.setInt(ANGEL_PSAGENT_CACHE_SYNC_TIMEINTERVAL_MS, 100000000)
    hadoopConf.set(ANGEL_LOG_PATH, tempPath)

    addUserResourceFiles(conf, hadoopConf)

    // Some other settings
    conf.getAllWithPrefix("spark.angel").foreach {
      case (key, value) => hadoopConf.set(s"angel$key", value)
    }
    hadoopConf
  }

  def main(args: Array[String]): Unit = {

    val path = Paths.get("./plasma_java").toAbsolutePath.toString
    val dir = new File(path)
    dir.listFiles().foreach(f => println(f))
    val input = Array("1","2")
    val numTask = args(0).toInt - 1


    val sconf = new SparkConf()
    val sc = new SparkContext(sconf)
    val hadoopConf = convertToHadoop(sc.getConf)

    val conf = Map(
      "plasma.store.path" -> "./plasma_java/plasma_java/bin/plasma_store_server",
      //"plasma.store.path" -> "/home/uuirs/anaconda3/envs/plasma_java/bin/plasma_store_server",
      "plasma.store.suffix" -> "/tmp/plasma",
      "plasma.store.memoryGB" -> "1",
      "python.script.name" -> "./adeepfm.py",
      "python.exec.path" -> "./torch/torch/bin/python"
      //"python.exec.path" -> "/home/uuirs/anaconda3/envs/torch/bin/python"
    )
    conf.foreach{
      case (key: String, value: String) =>
        hadoopConf.set(key, value)
      case _ =>
    }
    Master.embedStart(8980, numTask, AsyncModel.BSP, hadoopConf)

    val master_ip = InetAddress.getLocalHost.getHostAddress
    Worker.embedStart(master_ip, 8980, 9005)

    val StartTime = System.currentTimeMillis()
    if(numTask > 0) {
      val train = sc.parallelize(input).coalesce(numTask)
      train.mapPartitionsWithIndex {
        case (idx, iterator) => {

          val writer = new PrintWriter((new File("./input_" + idx)))
          for (x <- iterator) {
            writer.write(x)
          }
          writer.close()
          Worker.start(master_ip, 8980, 9006 + idx)
          Seq(1).toIterator
        }
      }.collect()
    }
    Thread.sleep(60*1000)

    Master.embedStop()
    Worker.embedStop()
    sc.stop()
  }

}
