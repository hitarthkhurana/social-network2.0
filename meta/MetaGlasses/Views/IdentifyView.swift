import SwiftUI

/// One-tap face identification.
/// Captures a single JPEG frame → POST /identify → TTS speaks name + details.
struct IdentifyView: View {
    @EnvironmentObject var glassesManager: GlassesManager

    enum State {
        case idle, capturing, waiting, result(IdentifyResult), error(String)
    }

    @SwiftUI.State private var state: State = .idle

    var body: some View {
        VStack(spacing: 24) {
            instructionText

            identifyButton

            resultCard
        }
        .padding(.top, 8)
    }

    // MARK: - Subviews

    private var instructionText: some View {
        Text("Point the glasses at someone, then tap Identify.")
            .font(.subheadline)
            .foregroundColor(.gray)
            .multilineTextAlignment(.center)
            .padding(.horizontal)
    }

    private var identifyButton: some View {
        Button {
            Task { await runIdentify() }
        } label: {
            HStack(spacing: 12) {
                if case .capturing = state {
                    ProgressView().tint(.white)
                } else if case .waiting = state {
                    ProgressView().tint(.white)
                } else {
                    Image(systemName: "person.fill.questionmark")
                        .font(.title2)
                }
                Text(buttonLabel)
                    .font(.headline)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 18)
            .background(buttonColor)
            .foregroundColor(.white)
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
        .disabled(isButtonDisabled)
    }

    @ViewBuilder
    private var resultCard: some View {
        switch state {
        case .result(let result):
            VStack(spacing: 8) {
                if result.isKnown {
                    Label(result.name ?? "", systemImage: "person.fill.checkmark")
                        .font(.title3.bold())
                        .foregroundColor(.green)
                    if let details = result.details, !details.isEmpty {
                        Text(details)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                } else {
                    Label("Person not recognized", systemImage: "questionmark.circle")
                        .font(.title3)
                        .foregroundColor(.orange)
                }
            }
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

    private func runIdentify() async {
        print("🔍 Identify: capturing photo...")
        state = .capturing

        guard let jpeg = await glassesManager.capturePhoto() else {
            print("❌ Identify: capture failed — stream not active?")
            state = .error("Could not capture photo — is the stream active?")
            return
        }

        print("🔍 Identify: sending \(jpeg.count) bytes to /identify...")
        state = .waiting

        do {
            let result = try await BackendClient.shared.identify(jpeg: jpeg)
            print("✅ Identify: name=\(result.name ?? "nil"), details=\(result.details ?? "nil")")
            state = .result(result)
            SpeechOutput.shared.speak(result.spokenDescription)
        } catch {
            print("❌ Identify: \(error)")
            state = .error("Backend error: \(error.localizedDescription)")
        }
    }

    // MARK: - Helpers

    private var buttonLabel: String {
        switch state {
        case .capturing: return "Capturing…"
        case .waiting:   return "Identifying…"
        default:         return "Identify"
        }
    }

    private var buttonColor: Color {
        switch state {
        case .capturing, .waiting: return .blue.opacity(0.6)
        default:                   return .blue
        }
    }

    private var isButtonDisabled: Bool {
        if case .capturing = state { return true }
        if case .waiting   = state { return true }
        return false
    }
}
