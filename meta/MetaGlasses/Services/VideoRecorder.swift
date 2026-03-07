import Foundation
import AVFoundation
import UIKit

/// Records glasses camera frames + HFP microphone audio into a single .mp4 file.
///
/// Strategy:
///  - Video: frames come from the DAT SDK via `appendFrame(_:)` calls.
///           Each UIImage is rendered into a CVPixelBuffer and appended to AVAssetWriter.
///  - Audio: recorded independently with AVAudioRecorder (simpler and more reliable than
///           AVAudioEngine taps for a hackathon timeline).
///  - At stopRecording(), the two files are merged with AVMutableComposition → final .mp4.
class VideoRecorder {

    // Output dimensions must match the DAT SDK's .low streaming resolution
    private let videoWidth  = 360
    private let videoHeight = 640
    private let videoFPS    = 7

    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?

    private var audioRecorder: AVAudioRecorder?
    private var audioURL: URL?
    private var videoURL: URL?

    private var isRecording = false
    private var startTime: CMTime?
    private var frameCount: Int64 = 0

    // MARK: - Start

    func startRecording() throws {
        guard !isRecording else { return }

        try setupAudioSession()

        videoURL = tempURL(extension: "mov")
        audioURL = tempURL(extension: "m4a")

        try startVideoWriter()
        try startAudioRecorder()

        isRecording = true
        frameCount = 0
        startTime = nil
    }

    // MARK: - Append video frame (called from GlassesManager.onVideoFrame)

    func appendFrame(_ image: UIImage) {
        guard
            isRecording,
            let adaptor = pixelBufferAdaptor,
            let vInput = videoInput,
            vInput.isReadyForMoreMediaData
        else { return }

        guard let pixelBuffer = image.toCVPixelBuffer(width: videoWidth, height: videoHeight)
        else { return }

        // Build a monotonic presentation timestamp at the target frame rate
        let pts = CMTime(value: frameCount, timescale: CMTimeScale(videoFPS))
        if startTime == nil { startTime = CMClockGetTime(CMClockGetHostTimeClock()) }

        adaptor.append(pixelBuffer, withPresentationTime: pts)
        frameCount += 1
    }

    // MARK: - Stop

    /// Stops recording, merges video + audio, returns the final .mp4 URL.
    func stopRecording() async -> URL? {
        guard isRecording else { return nil }
        isRecording = false

        audioRecorder?.stop()
        audioRecorder = nil

        videoInput?.markAsFinished()
        await assetWriter?.finishWriting()

        guard let vURL = videoURL, let aURL = audioURL else { return nil }

        let outputURL = tempURL(extension: "mp4")
        do {
            try await mergeVideoAndAudio(videoURL: vURL, audioURL: aURL, outputURL: outputURL)
            return outputURL
        } catch {
            print("Merge failed: \(error) — returning video-only file")
            return vURL
        }
    }

    // MARK: - Private setup

    private func setupAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        // .allowBluetooth routes mic input through HFP (the glasses mic)
        try session.setCategory(.playAndRecord, mode: .default, options: [.allowBluetooth, .defaultToSpeaker])
        try session.setActive(true, options: .notifyOthersOnDeactivation)
        // Give HFP time to fully connect before we start writing
        Thread.sleep(forTimeInterval: 2.0)
    }

    private func startVideoWriter() throws {
        guard let url = videoURL else { return }

        let writer = try AVAssetWriter(url: url, fileType: .mov)

        let videoSettings: [String: Any] = [
            AVVideoCodecKey:  AVVideoCodecType.h264,
            AVVideoWidthKey:  videoWidth,
            AVVideoHeightKey: videoHeight,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 500_000   // 500 kbps — fine for 360p over BT
            ]
        ]
        let vInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        vInput.expectsMediaDataInRealTime = true

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: vInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey  as String: videoWidth,
                kCVPixelBufferHeightKey as String: videoHeight
            ])

        writer.add(vInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        assetWriter = writer
        videoInput = vInput
        pixelBufferAdaptor = adaptor
    }

    private func startAudioRecorder() throws {
        guard let url = audioURL else { return }

        let settings: [String: Any] = [
            AVFormatIDKey:            kAudioFormatMPEG4AAC,
            AVSampleRateKey:          8000,     // HFP streams at 8 kHz
            AVNumberOfChannelsKey:    1,
            AVEncoderBitRateKey:      16_000
        ]
        let recorder = try AVAudioRecorder(url: url, settings: settings)
        recorder.record()
        audioRecorder = recorder
    }

    // MARK: - Merge

    private func mergeVideoAndAudio(videoURL: URL, audioURL: URL, outputURL: URL) async throws {
        let videoAsset = AVAsset(url: videoURL)
        let audioAsset = AVAsset(url: audioURL)

        let composition = AVMutableComposition()

        let videoDuration = try await videoAsset.load(.duration)
        let audioDuration = try await audioAsset.load(.duration)
        let duration      = min(videoDuration, audioDuration)
        let timeRange     = CMTimeRange(start: .zero, duration: duration)

        if let srcVideo = try await videoAsset.loadTracks(withMediaType: .video).first {
            let compVideo = composition.addMutableTrack(
                withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid)
            try compVideo?.insertTimeRange(timeRange, of: srcVideo, at: .zero)
        }

        if let srcAudio = try await audioAsset.loadTracks(withMediaType: .audio).first {
            let compAudio = composition.addMutableTrack(
                withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid)
            try compAudio?.insertTimeRange(timeRange, of: srcAudio, at: .zero)
        }

        guard let exportSession = AVAssetExportSession(
            asset: composition,
            presetName: AVAssetExportPresetMediumQuality)
        else { throw RecorderError.exportSessionCreationFailed }

        exportSession.outputURL      = outputURL
        exportSession.outputFileType = .mp4
        await exportSession.export()

        if let error = exportSession.error {
            throw error
        }
    }

    // MARK: - Helpers

    private func tempURL(extension ext: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension(ext)
    }
}

enum RecorderError: Error {
    case exportSessionCreationFailed
}
