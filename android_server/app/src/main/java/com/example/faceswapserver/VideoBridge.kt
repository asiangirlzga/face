package com.example.faceswapserver

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.os.Build
import android.view.Surface
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import kotlin.math.roundToInt

/**
 * Video decode/encode for the Python server (server._AndroidVideo).
 * Reader: MediaMetadataRetriever returns frame i as JPEG.
 * Writer: JPEG frames are drawn onto a MediaCodec H.264 input surface and muxed into an MP4,
 * with the source's audio track copied across when the container allows it.
 */
object VideoBridge {
    // ---------------------------------------------------------------- reader
    private var retriever: MediaMetadataRetriever? = null
    private var fps = 30.0

    /** Returns [frameCount, fps * 1000]. */
    @JvmStatic
    @Synchronized
    fun openReader(path: String): IntArray {
        closeReader()
        val r = MediaMetadataRetriever()
        r.setDataSource(path)
        retriever = r
        val durMs = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
        var frames = 0
        if (Build.VERSION.SDK_INT >= 28) {
            frames = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull() ?: 0
        }
        var rate = 0.0
        if (frames > 0 && durMs > 0) {
            rate = frames * 1000.0 / durMs
        } else {
            rate = trackFrameRate(path)
            if (rate <= 0) rate = 30.0
            frames = (durMs * rate / 1000.0).toInt()
        }
        fps = rate
        return intArrayOf(frames, (rate * 1000).roundToInt())
    }

    private fun trackFrameRate(path: String): Double {
        val ex = MediaExtractor()
        try {
            ex.setDataSource(path)
            for (i in 0 until ex.trackCount) {
                val f = ex.getTrackFormat(i)
                if (f.getString(MediaFormat.KEY_MIME)?.startsWith("video/") == true &&
                    f.containsKey(MediaFormat.KEY_FRAME_RATE)
                ) {
                    return runCatching { f.getInteger(MediaFormat.KEY_FRAME_RATE).toDouble() }
                        .getOrElse { f.getFloat(MediaFormat.KEY_FRAME_RATE).toDouble() }
                }
            }
        } catch (_: Exception) {
        } finally {
            ex.release()
        }
        return 0.0
    }

    @JvmStatic
    @Synchronized
    fun readFrame(index: Int): ByteArray? {
        val r = retriever ?: return null
        val us = (index * 1_000_000.0 / fps).toLong()
        val bmp = r.getFrameAtTime(us, MediaMetadataRetriever.OPTION_CLOSEST) ?: return null
        val out = ByteArrayOutputStream(bmp.byteCount / 8)
        bmp.compress(Bitmap.CompressFormat.JPEG, 92, out)
        bmp.recycle()
        return out.toByteArray()
    }

    @JvmStatic
    @Synchronized
    fun closeReader() {
        runCatching { retriever?.release() }
        retriever = null
    }

    // ---------------------------------------------------------------- writer
    private var enc: MediaCodec? = null
    private var surface: Surface? = null
    private var muxer: MediaMuxer? = null
    private var videoTrack = -1
    private var audioTrack = -1
    private var audioEx: MediaExtractor? = null
    private var muxerStarted = false
    private var outFrames = 0L
    private var outFps = 30.0
    private var w = 0
    private var h = 0
    private var srcPath: String? = null

    @JvmStatic
    @Synchronized
    fun openWriter(path: String, width: Int, height: Int, rate: Double, source: String?) {
        w = width; h = height; outFps = rate; srcPath = source
        outFrames = 0; videoTrack = -1; audioTrack = -1; muxerStarted = false
        val fmt = MediaFormat.createVideoFormat("video/avc", w, h).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface)
            setInteger(MediaFormat.KEY_BIT_RATE, (w.toDouble() * h * rate * 0.12).toInt().coerceIn(1_500_000, 24_000_000))
            setInteger(MediaFormat.KEY_FRAME_RATE, rate.roundToInt().coerceIn(1, 60))
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        val codec = MediaCodec.createEncoderByType("video/avc")
        codec.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        surface = codec.createInputSurface()
        codec.start()
        enc = codec
        muxer = MediaMuxer(path, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    }

    @JvmStatic
    @Synchronized
    fun writeFrame(jpeg: ByteArray) {
        val bmp = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size) ?: return
        val s = surface ?: return
        val canvas = if (Build.VERSION.SDK_INT >= 26) {
            runCatching { s.lockHardwareCanvas() }.getOrElse { s.lockCanvas(null) }
        } else s.lockCanvas(null)
        canvas.drawBitmap(bmp, null, Rect(0, 0, w, h), null)
        s.unlockCanvasAndPost(canvas)
        bmp.recycle()
        drain(false)
    }

    @JvmStatic
    @Synchronized
    fun closeWriter() {
        val codec = enc ?: return
        try {
            codec.signalEndOfInputStream()
            drain(true)
            copyAudio()
        } finally {
            runCatching { codec.stop() }
            runCatching { codec.release() }
            runCatching { surface?.release() }
            runCatching { if (muxerStarted) muxer?.stop() }
            runCatching { muxer?.release() }
            runCatching { audioEx?.release() }
            enc = null; surface = null; muxer = null; audioEx = null; muxerStarted = false
        }
    }

    private fun startMuxer(codec: MediaCodec) {
        val m = muxer ?: return
        videoTrack = m.addTrack(codec.outputFormat)
        val src = srcPath
        if (src != null) {
            try {
                val ex = MediaExtractor()
                ex.setDataSource(src)
                for (i in 0 until ex.trackCount) {
                    val f = ex.getTrackFormat(i)
                    if (f.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                        audioTrack = m.addTrack(f)   // throws if the MP4 muxer cannot take this codec
                        ex.selectTrack(i)
                        audioEx = ex
                        break
                    }
                }
                if (audioEx == null) ex.release()
            } catch (_: Exception) {
                audioTrack = -1
                runCatching { audioEx?.release() }
                audioEx = null
            }
        }
        m.start()
        muxerStarted = true
    }

    private fun drain(endOfStream: Boolean) {
        val codec = enc ?: return
        val info = MediaCodec.BufferInfo()
        while (true) {
            val idx = codec.dequeueOutputBuffer(info, if (endOfStream) 10_000 else 0)
            when {
                idx == MediaCodec.INFO_TRY_AGAIN_LATER -> if (!endOfStream) return
                idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> if (!muxerStarted) startMuxer(codec)
                idx >= 0 -> {
                    val buf = codec.getOutputBuffer(idx)
                    if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) info.size = 0
                    if (info.size > 0 && muxerStarted && buf != null) {
                        buf.position(info.offset)
                        buf.limit(info.offset + info.size)
                        // Frames were drawn at processing speed; rewrite to the source frame rate.
                        info.presentationTimeUs = (outFrames * 1_000_000.0 / outFps).toLong()
                        outFrames++
                        muxer!!.writeSampleData(videoTrack, buf, info)
                    }
                    codec.releaseOutputBuffer(idx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
            }
        }
    }

    private fun copyAudio() {
        val ex = audioEx ?: return
        val m = muxer ?: return
        if (audioTrack < 0 || !muxerStarted) return
        val limitUs = (outFrames * 1_000_000.0 / outFps).toLong()
        val buf = ByteBuffer.allocate(1 shl 20)
        val info = MediaCodec.BufferInfo()
        while (true) {
            buf.clear()
            val n = ex.readSampleData(buf, 0)
            val t = ex.sampleTime
            if (n < 0 || t < 0 || t > limitUs) break
            val flags = if (ex.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0) MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
            info.set(0, n, t, flags)
            m.writeSampleData(audioTrack, buf, info)
            ex.advance()
        }
    }
}
