import AVFoundation

/// Speaks text through the connected glasses speakers via Bluetooth A2DP.
/// When an HFP/A2DP Bluetooth device (the glasses) is active, iOS automatically
/// routes AVSpeechSynthesizer output to it.
class SpeechOutput: NSObject, AVSpeechSynthesizerDelegate {
    static let shared = SpeechOutput()

    private let synthesizer = AVSpeechSynthesizer()
    private var completionHandler: (() -> Void)?

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    /// Speaks the given text. Calls completion when the utterance finishes.
    func speak(_ text: String, completion: (() -> Void)? = nil) {
        synthesizer.stopSpeaking(at: .immediate)
        completionHandler = completion

        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0
        // Use a clear, neutral voice
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")

        synthesizer.speak(utterance)
    }

    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }

    // MARK: - AVSpeechSynthesizerDelegate

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                            didFinish utterance: AVSpeechUtterance) {
        completionHandler?()
        completionHandler = nil
    }
}
