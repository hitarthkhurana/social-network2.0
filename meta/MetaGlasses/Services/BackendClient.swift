import Foundation

// MARK: - Response models

struct IdentifyResult: Codable {
    let name: String?
    let details: String?

    var isKnown: Bool { name != nil }

    var spokenDescription: String {
        guard let name else { return "Person not recognized." }
        if let details, !details.isEmpty {
            return "\(name). \(details)"
        }
        return name
    }
}

struct EnrollResult: Codable {
    let status: String
    let name: String?
    let message: String?
}

// MARK: - Client

class BackendClient {
    static let shared = BackendClient()

    /// Point this at your team's server. Can be changed at runtime from the UI.
    var baseURL: String = "http://localhost:8000"

    // MARK: - Identify (fast path — single JPEG frame)

    /// Sends one JPEG image to the backend for face recognition.
    /// Expected response time: ~1-3 seconds.
    func identify(jpeg: Data) async throws -> IdentifyResult {
        let url = try endpoint("/identify")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("image/jpeg", forHTTPHeaderField: "Content-Type")
        request.httpBody = jpeg
        request.timeoutInterval = 10

        let (data, response) = try await URLSession.shared.data(for: request)
        try validate(response)
        return try JSONDecoder().decode(IdentifyResult.self, from: data)
    }

    // MARK: - Enroll (video clip — name + details extracted by backend)

    /// Uploads a .mp4 clip to the backend for face embedding + audio transcription.
    func enroll(videoURL: URL) async throws -> EnrollResult {
        let url = try endpoint("/enroll")
        let boundary = "boundary-\(UUID().uuidString)"

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 60  // video uploads can take a few seconds

        let videoData = try Data(contentsOf: videoURL)
        request.httpBody = multipartBody(data: videoData,
                                         boundary: boundary,
                                         fieldName: "video",
                                         filename: "clip.mp4",
                                         mimeType: "video/mp4")

        let (data, response) = try await URLSession.shared.data(for: request)
        try validate(response)
        return try JSONDecoder().decode(EnrollResult.self, from: data)
    }

    // MARK: - Helpers

    private func endpoint(_ path: String) throws -> URL {
        guard let url = URL(string: baseURL + path) else {
            throw BackendError.invalidURL(baseURL + path)
        }
        return url
    }

    private func validate(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw BackendError.httpError(http.statusCode)
        }
    }

    private func multipartBody(data: Data,
                                boundary: String,
                                fieldName: String,
                                filename: String,
                                mimeType: String) -> Data {
        var body = Data()
        let crlf = "\r\n"
        body += "--\(boundary)\(crlf)".utf8Data
        body += "Content-Disposition: form-data; name=\"\(fieldName)\"; filename=\"\(filename)\"\(crlf)".utf8Data
        body += "Content-Type: \(mimeType)\(crlf)\(crlf)".utf8Data
        body += data
        body += "\(crlf)--\(boundary)--\(crlf)".utf8Data
        return body
    }
}

// MARK: - Errors

enum BackendError: LocalizedError {
    case invalidURL(String)
    case httpError(Int)

    var errorDescription: String? {
        switch self {
        case .invalidURL(let url): return "Invalid backend URL: \(url)"
        case .httpError(let code): return "Backend returned HTTP \(code)"
        }
    }
}

// MARK: - Convenience

private extension String {
    var utf8Data: Data { Data(utf8) }
}
