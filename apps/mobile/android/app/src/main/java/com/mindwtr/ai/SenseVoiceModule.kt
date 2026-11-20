package com.mindwtr.ai

import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaCodec
import com.facebook.react.bridge.*
import com.k2fsa.sherpa.onnx.*
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SenseVoiceModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private var recognizer: OfflineRecognizer? = null

    override fun getName(): String = "SenseVoiceModule"

    @ReactMethod
    fun initModel(modelDir: String, promise: Promise) {
        try {
            val modelPath = "$modelDir/model.int8.onnx"
            val tokensPath = "$modelDir/tokens.txt"

            if (!File(modelPath).exists() || !File(tokensPath).exists()) {
                promise.reject("MODEL_NOT_FOUND", "SenseVoice model files missing in $modelDir")
                return
            }

            val config = OfflineRecognizerConfig(
                model = OfflineModelConfig(
                    senseVoice = OfflineSenseVoiceModelConfig(
                        model = modelPath,
                        language = "auto", // auto-detect zh, en, ja, ko, yue
                        useItn = true     // Convert "twenty three" -> "23"
                    ),
                    tokens = tokensPath,
                    numThreads = 4,
                    provider = "cpu"
                )
            )
            recognizer = OfflineRecognizer(reactContext.assets, config)
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("INIT_ERROR", e.localizedMessage, e)
        }
    }

    @ReactMethod
    fun transcribeFile(audioPath: String, promise: Promise) {
        val rec = recognizer ?: run {
            promise.reject("NOT_INITIALIZED", "Call initModel first")
            return
        }

        Thread {
            try {
                // Decode audio file (m4a/wav) to 16kHz mono PCM FloatArray
                val samples = decodeAudioTo16kMono(audioPath)
                val stream = rec.createStream()
                stream.acceptWaveform(samples, 16000)
                rec.decode(stream)
                val result = rec.getResult(stream)
                stream.release()

                promise.resolve(result.text.trim())
            } catch (e: Exception) {
                promise.reject("TRANSCRIBE_ERROR", e.localizedMessage, e)
            }
        }.start()
    }

    // Decodes audio using Android MediaCodec into 16kHz float PCM samples
    private fun decodeAudioTo16kMono(filePath: String): FloatArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(filePath)
        val format = extractor.getTrackFormat(0)
        val mime = format.getString(MediaFormat.KEY_MIME) ?: "audio/mp4a-latm"

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val pcmBytes = mutableListOf<Byte>()
        val info = MediaCodec.BufferInfo()
        var isEOS = false

        extractor.selectTrack(0)
        while (!isEOS) {
            val inIndex = codec.dequeueInputBuffer(10000)
            if (inIndex >= 0) {
                val buffer = codec.getInputBuffer(inIndex)!!
                val sampleSize = extractor.readSampleData(buffer, 0)
                if (sampleSize < 0) {
                    codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                    isEOS = true
                } else {
                    codec.queueInputBuffer(inIndex, 0, sampleSize, extractor.sampleTime, 0)
                    extractor.advance()
                }
            }

            var outIndex = codec.dequeueOutputBuffer(info, 10000)
            while (outIndex >= 0) {
                val outBuffer = codec.getOutputBuffer(outIndex)!!
                val chunk = ByteArray(info.size)
                outBuffer.get(chunk)
                outBuffer.clear()
                for (b in chunk) pcmBytes.add(b)
                codec.releaseOutputBuffer(outIndex, false)
                outIndex = codec.dequeueOutputBuffer(info, 0)
            }
        }
        codec.stop()
        codec.release()
        extractor.release()

        // Convert 16-bit PCM bytes to FloatArray (-1.0 to 1.0)
        val shortBuffer = ByteBuffer.wrap(pcmBytes.toByteArray())
            .order(ByteOrder.LITTLE_ENDIAN)
            .asShortBuffer()
        val floats = FloatArray(shortBuffer.remaining())
        for (i in floats.indices) {
            floats[i] = shortBuffer.get(i) / 32768.0f
        }
        return floats
    }
}
