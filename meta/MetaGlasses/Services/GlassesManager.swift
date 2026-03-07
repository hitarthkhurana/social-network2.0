import Foundation
import SwiftUI
import MWDATCore
import MWDATCamera

/// Central manager for all Meta Wearables DAT SDK interactions.
/// Handles registration, camera streaming, and photo capture.
@MainActor
class GlassesManager: ObservableObject {

    // MARK: - Published state (drives UI)

    @Published var isRegistered = false
    @Published var isStreaming = false
    @Published var latestFrame: UIImage?
    @Published var statusMessage = "Not connected"

    // MARK: - Private

    private let wearables = Wearables.shared
    private var streamSession: StreamSession?

    // Listener tokens must be retained or subscriptions die
    private var stateToken: Any?
    private var frameToken: Any?
    private var photoToken: Any?

    // One-shot continuation for capturePhoto()
    private var photoContinuation: CheckedContinuation<Data?, Never>?

    // Callback forwarded to VideoRecorder while enrolling
    var onVideoFrame: ((UIImage) -> Void)?

    // MARK: - Registration

    func register() async {
        print("🔵 Starting registration...")

        // Check current state first
        for await state in wearables.registrationStateStream() {
            print("📱 Current registration state: \(state) (raw: \(state.rawValue))")
            if state == .registered {
                print("✅ Already registered! Skipping registration, requesting camera...")
                isRegistered = true
                statusMessage = "Already registered"
                await requestCameraPermission()
                return
            }
            break
        }

        do {
            let result = try await wearables.startRegistration()
            statusMessage = "Opening Meta AI app for registration…"
            print("✅ Registration result: \(result)")
        } catch {
            print("❌ Registration error: \(error)")
            print("❌ Error details: \(String(describing: error))")
            statusMessage = "Registration error: \(error.localizedDescription)"
        }
    }

    /// Called from .onOpenURL in the app entry point so Meta AI can return to this app
    func handleCallback(url: URL) async throws {
        _ = try await wearables.handleUrl(url)
    }

    func unregister() async {
        do {
            try await wearables.startUnregistration()
            isRegistered = false
            statusMessage = "Unregistered. Tap Connect to re-register."
            print("✅ Unregistration complete")
        } catch {
            print("❌ Unregistration error: \(error)")
            statusMessage = "Unregistration error: \(error.localizedDescription)"
        }
    }

    func requestCameraPermission() async {
        // First check if we already have permission
        do {
            let currentStatus = try await wearables.checkPermissionStatus(.camera)
            print("📸 Current camera permission status: \(currentStatus)")
            if currentStatus == .granted {
                statusMessage = "Camera permission already granted!"
                return
            }
        } catch {
            print("⚠️ Check permission error: \(error)")
        }

        // Request permission if not already granted
        do {
            print("📸 Requesting camera permission...")
            let status = try await wearables.requestPermission(.camera)
            print("📸 Camera permission result: \(status)")
            statusMessage = "Camera permission: \(status)"
        } catch {
            print("❌ Permission request error: \(error)")
            print("❌ Error details: \(String(describing: error))")
            statusMessage = "Permission error: \(error.localizedDescription)"
        }
    }

    // MARK: - Streaming

    /// Starts the camera stream if registration has already been completed.
    func startStreamIfReady() {
        // Observe registration state
        Task {
            for await state in wearables.registrationStateStream() {
                print("📱 Registration state changed: \(state)")
                isRegistered = (state == .registered)
                if isRegistered && !isStreaming {
                    print("✅ Glasses are registered! Starting stream...")
                    await startStream()
                } else if !isRegistered {
                    print("⚠️ Glasses not registered yet. State: \(state)")
                }
            }
        }
    }

    func startStream() async {
        guard !isStreaming else { return }

        // Low resolution + 7 fps = best reliability over Bluetooth Classic bandwidth
        let config = StreamSessionConfig(
            videoCodec: .raw,
            resolution: .low,   // 360 x 640
            frameRate: 7)

        let session = StreamSession(
            streamSessionConfig: config,
            deviceSelector: AutoDeviceSelector(wearables: wearables))

        stateToken = session.statePublisher.listen { [weak self] state in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.isStreaming = (state == .streaming)
                self.statusMessage = stateDescription(state)
            }
        }

        frameToken = session.videoFramePublisher.listen { [weak self] frame in
            guard let image = frame.makeUIImage() else { return }
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.latestFrame = image
                self.onVideoFrame?(image)
            }
        }

        // Single photo callback — fires once per capturePhoto() call
        photoToken = session.photoDataPublisher.listen { [weak self] photoData in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.photoContinuation?.resume(returning: photoData.data)
                self.photoContinuation = nil
            }
        }

        streamSession = session
        statusMessage = "Connecting to glasses…"
        await session.start()
    }

    func stopStream() async {
        await streamSession?.stop()
        streamSession = nil
        stateToken = nil
        frameToken = nil
        photoToken = nil
        isStreaming = false
        statusMessage = "Stream stopped"
    }

    // MARK: - Photo capture

    /// Captures one JPEG frame from the active stream.
    /// Returns nil if no stream is active or capture times out.
    func capturePhoto(timeoutSeconds: Double = 5) async -> Data? {
        guard let session = streamSession else {
            statusMessage = "No active stream — cannot capture photo"
            return nil
        }

        return await withCheckedContinuation { continuation in
            photoContinuation = continuation
            session.capturePhoto(format: .jpeg)

            // Safety timeout so we never hang indefinitely
            Task {
                try? await Task.sleep(nanoseconds: UInt64(timeoutSeconds * 1_000_000_000))
                if self.photoContinuation != nil {
                    self.photoContinuation?.resume(returning: nil)
                    self.photoContinuation = nil
                }
            }
        }
    }
}

// MARK: - Helpers

private func stateDescription(_ state: StreamSessionState) -> String {
    switch state {
    case .waitingForDevice: return "Waiting for glasses…"
    case .starting:         return "Starting stream…"
    case .streaming:        return "Streaming"
    case .paused:           return "Paused"
    case .stopping:         return "Stopping…"
    case .stopped:          return "Stopped"
    @unknown default:       return "Unknown"
    }
}
