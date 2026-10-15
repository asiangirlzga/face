package com.example.faceswapserver

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap

/**
 * ONNX Runtime for the Python server (engine.AndroidOrt). Tensors cross the bridge as
 * little-endian float32 bytes plus a long[] shape, outputs come back as float32 bytes.
 */
object OrtBridge {
    private val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }
    private val sessions = ConcurrentHashMap<String, OrtSession>()

    @JvmStatic
    @Synchronized
    fun load(name: String, path: String) {
        sessions.remove(name)?.close()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(2, 6))
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        sessions[name] = env.createSession(path, opts)
    }

    @JvmStatic
    fun run1(name: String, d0: ByteArray, s0: LongArray): Array<ByteArray> =
        run(name, listOf(d0 to s0))

    @JvmStatic
    fun run2(name: String, d0: ByteArray, s0: LongArray, d1: ByteArray, s1: LongArray): Array<ByteArray> =
        run(name, listOf(d0 to s0, d1 to s1))

    private fun run(name: String, inputs: List<Pair<ByteArray, LongArray>>): Array<ByteArray> {
        val session = sessions[name] ?: error("Model not loaded: $name")
        val inputNames = session.inputNames.toList()
        val tensors = ArrayList<OnnxTensor>()
        try {
            val feed = HashMap<String, OnnxTensor>()
            inputs.forEachIndexed { i, (data, shape) ->
                val bb = ByteBuffer.allocateDirect(data.size).order(ByteOrder.LITTLE_ENDIAN)
                bb.put(data)
                bb.rewind()
                val t = OnnxTensor.createTensor(env, bb.asFloatBuffer(), shape)
                tensors.add(t)
                feed[inputNames[i]] = t
            }
            session.run(feed).use { result ->
                return Array(result.size()) { i ->
                    val fb = (result.get(i) as OnnxTensor).floatBuffer
                    val out = ByteBuffer.allocate(fb.remaining() * 4).order(ByteOrder.LITTLE_ENDIAN)
                    out.asFloatBuffer().put(fb)
                    out.array()
                }
            }
        } finally {
            tensors.forEach { it.close() }
        }
    }
}
