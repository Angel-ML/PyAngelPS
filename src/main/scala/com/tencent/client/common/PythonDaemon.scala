package com.tencent.client.common

import java.io.DataOutputStream
import java.util.logging.Logger

import scala.sys.process._
class PythonDaemon {

}

object PythonDaemon {

  private val logger = Logger.getLogger(PythonDaemon.getClass.getSimpleName)

  private var storeProcess: Process = _

  private var procOutput:

  private val daemonModule = "pyangel.daemon"

  def startDaemon(pythonExec: String, jvmPort: Int, userScriptPath: String, plasmaName: String): Unit ={
    if (storeProcess != null)
      return
    val envVars = Map(
      "jvm_port"->jvmPort.toString,
      "user_script"->userScriptPath,
      "plasma_name"->plasmaName
    )
    
    val procIO = new ProcessIO(
      in => {
        procOutput = new scala.io.Source.(in)
      },
      _.close(),
      _.close()
    )

    storeProcess = Process(Seq(pythonExec, "-m", daemonModule), None, envVars.toSeq: _*).run(procIO)

  }

  def createPythonProcess(): Unit ={

    procOutput
  }
}
