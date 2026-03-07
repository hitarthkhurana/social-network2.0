import SwiftUI

/// Enrollment flow: tap Record, look at the person and say their name + details,
/// tap Stop → .mp4 clip uploads to POST /enroll.
struct EnrollView: View {
    @EnvironmentObject var glassesManager: GlassesManager

    enum State {
        case idle, recording, uploading, success(String), error(String)
    }

    @SwiftUI.State private var state: State = .idle
    @SwiftUI.State private var elapsedSeconds = 0
    @SwiftUI.State private var recordingTimer: Timer?

    private let recorder = VideoRecorder()
    private let maxRecordingSeconds = 20

    var body: some View {
        VStack(spacing: 24) {
            instructionText

            recordButton

            if case .recording = state {
                recordingIndicator
            }

            resultCard
        }
        .padding(.top, 8)
        .onDisappear {
            // Safety: stop any ongoing recording if user switches tabs
            if case .recording = state {
                Task { await stopAndUpload() }
            }
        }
    }

    // MARK: - Subviews

    private var instructionText: some View {
        VStack(spacing: 4) {
            Text("Look at the person and tap Record.")
                .font(.subheadline)
                .foregroundColor(.gray)
            Text("Say their name and one or two details out loud.")
                .font(.caption)
                .foregroundColor(.gray.opacity(0.7))
        }
        .multilineTextAlignment(.center)
        .padding(.horizontal)
    }

    private var recordButton: some View {
        Button {
            Task {
                if case .recording = state {
                    await stopAndUpload()
                } else {
                    await startRecording()
                }
            }
        } label: {
            HStack(spacing: 12) {
                if case .uploading = state {
                    ProgressView().tint(.white)
                } else {
                    Image(systemName: recordingIcon)
                        .font(.title2)
                }
                Text(recordingLabel)
                    .font(.headline)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 18)
            .background(recordingColor)
            .foregroundColor(.white)
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
        .disabled({ if case .uploading = state { return true }; return false }())
    }

    private var recordingIndicator: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(Color.red)
                .frame(width: 10, height: 10)
                .opacity(elapsedSeconds % 2 == 0 ? 1 : 0.3)
                .animation(.easeInOut(duration: 0.5).repeatForever(), value: elapsedSeconds)
            Text("\(elapsedSeconds)s / \(maxRecordingSeconds)s")
                .font(.caption.monospacedDigit())
                .foregroundColor(.red)
            Spacer()
            Text("Recording — tap Stop when done")
                .font(.caption)
                .foregroundColor(.gray)
        }
        .padding(.horizontal, 4)
    }

    @ViewBuilder
    private var resultCard: some View {
        switch state {
        case .success(let name):
            Label("\(name) saved!", systemImage: "person.badge.plus")
                .font(.title3.bold())
                .foregroundColor(.green)
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color.white.opacity(0.06))
                .clipShape(RoundedRectangle(cornerRadius: 14))

        case .error(let msg):
            Label(msg, systemImage: "exclamationmark.triangle")
                .font(.subheadline)
                .foregroundColor(.red)
                .multilineTextAlignment(.center)
                .padding()

        default:
            EmptyView()
        }
    }

    // MARK: - Actions

    private func startRecording() async {
        do {
            try recorder.startRecording()
        } catch {
            state = .error("Could not start recording: \(error.localizedDescription)")
            return
        }

        // Forward each glasses frame into the recorder
        glassesManager.onVideoFrame = { [weak recorder] image in
            recorder?.appendFrame(image)
        }

        state = .recording
        elapsedSeconds = 0

        // Update elapsed counter and auto-stop at max duration
        recordingTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            Task { @MainActor in
                self.elapsedSeconds += 1
                if self.elapsedSeconds >= self.maxRecordingSeconds {
                    await self.stopAndUpload()
                }
            }
        }
    }

    private func stopAndUpload() async {
        recordingTimer?.invalidate()
        recordingTimer = nil
        glassesManager.onVideoFrame = nil

        state = .uploading

        guard let videoURL = await recorder.stopRecording() else {
            state = .error("Recording produced no output.")
            return
        }

        do {
            let result = try await BackendClient.shared.enroll(videoURL: videoURL)
            let name = result.name ?? "Person"
            state = .success(name)
            SpeechOutput.shared.speak("\(name) has been saved.")
        } catch {
            state = .error("Upload failed: \(error.localizedDescription)")
        }

        // Clean up temp file
        try? FileManager.default.removeItem(at: videoURL)
    }

    // MARK: - Helpers

    private var recordingIcon: String {
        if case .recording = state { return "stop.circle.fill" }
        return "record.circle"
    }

    private var recordingLabel: String {
        switch state {
        case .recording: return "Stop"
        case .uploading: return "Uploading…"
        default:         return "Record"
        }
    }

    private var recordingColor: Color {
        switch state {
        case .recording: return .red
        case .uploading: return .orange.opacity(0.7)
        default:         return .indigo
        }
    }
}
