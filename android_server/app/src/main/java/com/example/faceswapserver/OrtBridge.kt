package com.example.faceswapserver

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
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
            val result = session.run(feed)
            try {
                return Array(result.size()) { i -> toFloatBytes(result.get(i) as OnnxTensor) }
            } finally {
                result.close()
            }
        } finally {
            tensors.forEach { it.close() }
        }
    }

    // ---------------------------------------------------------------- dtype-aware API (Stable Diffusion, sdgen/pipeline.py)
    // Python feeds named inputs as raw little-endian bytes tagged with a dtype ("float32", "float16",
    // "int64", "int32"); every output comes back as float32 bytes.

    private val feeds = LinkedHashMap<String, Triple<String, ByteArray, LongArray>>()

    /** optLevel: "none" keeps big external-data weights memory-mapped, "basic"/"all" enable graph optimisation. */
    @JvmStatic
    @Synchronized
    fun loadModel(name: String, path: String, optLevel: String) {
        sessions.remove(name)?.close()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(2, 6))
            setOptimizationLevel(
                when (optLevel) {
                    "none" -> OrtSession.SessionOptions.OptLevel.NO_OPT
                    "basic" -> OrtSession.SessionOptions.OptLevel.BASIC_OPT
                    else -> OrtSession.SessionOptions.OptLevel.ALL_OPT
                }
            )
            setCPUArenaAllocator(false)
            setMemoryPatternOptimization(false)
        }
        sessions[name] = env.createSession(path, opts)
    }

    @JvmStatic
    @Synchronized
    fun unload(name: String) {
        sessions.remove(name)?.close()
        System.gc()
    }

    @JvmStatic
    fun inputNames(name: String): Array<String> =
        (sessions[name] ?: error("Model not loaded: $name")).inputNames.toTypedArray()

    @JvmStatic
    fun inputTypes(name: String): Array<String> {
        val s = sessions[name] ?: error("Model not loaded: $name")
        return s.inputNames.map { n -> typeName((s.inputInfo[n]!!.info as TensorInfo).type) }.toTypedArray()
    }

    @JvmStatic
    fun outputNames(name: String): Array<String> =
        (sessions[name] ?: error("Model not loaded: $name")).outputNames.toTypedArray()

    @JvmStatic
    @Synchronized
    fun feedClear() {
        feeds.clear()
    }

    @JvmStatic
    @Synchronized
    fun feed(inputName: String, dtype: String, data: ByteArray, shape: LongArray) {
        feeds[inputName] = Triple(dtype, data, shape)
    }

    @JvmStatic
    @Synchronized
    fun runFeed(name: String): Array<ByteArray> {
        val session = sessions[name] ?: error("Model not loaded: $name")
        val tensors = ArrayList<OnnxTensor>()
        try {
            val inputs = HashMap<String, OnnxTensor>()
            for ((key, v) in feeds) {
                val (dtype, data, shape) = v
                val bb = ByteBuffer.allocateDirect(data.size).order(ByteOrder.LITTLE_ENDIAN)
                bb.put(data)
                bb.rewind()
                val t = when (dtype) {
                    "float32" -> OnnxTensor.createTensor(env, bb.asFloatBuffer(), shape)
                    "float16" -> OnnxTensor.createTensor(env, bb, shape, OnnxJavaType.FLOAT16)
                    "int64" -> OnnxTensor.createTensor(env, bb.asLongBuffer(), shape)
                    "int32" -> OnnxTensor.createTensor(env, bb.asIntBuffer(), shape)
                    else -> error("Unsupported input type: $dtype")
                }
                tensors.add(t)
                inputs[key] = t
            }
            feeds.clear()
            val result = session.run(inputs)
            try {
                return Array(result.size()) { i -> toFloatBytes(result.get(i) as OnnxTensor) }
            } finally {
                result.close()
            }
        } finally {
            tensors.forEach { it.close() }
        }
    }

    private fun typeName(t: OnnxJavaType): String = when (t) {
        OnnxJavaType.FLOAT -> "float32"
        OnnxJavaType.FLOAT16 -> "float16"
        OnnxJavaType.INT64 -> "int64"
        OnnxJavaType.INT32 -> "int32"
        else -> t.name.lowercase()
    }

    private fun toFloatBytes(t: OnnxTensor): ByteArray {
        when (t.info.type) {
            OnnxJavaType.FLOAT -> {
                val fb = t.floatBuffer
                val out = ByteBuffer.allocate(fb.remaining() * 4).order(ByteOrder.LITTLE_ENDIAN)
                out.asFloatBuffer().put(fb)
                return out.array()
            }
            OnnxJavaType.FLOAT16 -> {
                val sb = t.byteBuffer.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                val n = sb.remaining()
                val out = ByteBuffer.allocate(n * 4).order(ByteOrder.LITTLE_ENDIAN)
                val fo = out.asFloatBuffer()
                for (i in 0 until n) fo.put(halfToFloat(sb.get(i).toInt() and 0xFFFF))
                return out.array()
            }
            else -> error("Unsupported output type: ${t.info.type}")
        }
    }

    private fun halfToFloat(h: Int): Float {
        val sign = if ((h shr 15) and 1 == 1) -1f else 1f
        val e = (h shr 10) and 0x1F
        val m = h and 0x3FF
        return when (e) {
            0 -> sign * Math.scalb(m.toFloat(), -24)
            31 -> if (m == 0) sign * Float.POSITIVE_INFINITY else Float.NaN
            else -> sign * Math.scalb(1f + m / 1024f, e - 15)
        }
    }
}
